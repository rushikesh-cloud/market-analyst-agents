from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import psycopg


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return dsn


def _conn_string() -> str:
    return _normalize_pg_dsn(_env("PGVECTOR_CONNECTION_STRING"))


def init_supervisor_chat_tables() -> None:
    sql_sessions = """
        CREATE TABLE IF NOT EXISTS supervisor_chat_sessions (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT NOT NULL,
            symbol TEXT,
            company TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    sql_sessions_compat = """
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS id TEXT;
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS session_id TEXT;
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS title TEXT;
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS symbol TEXT;
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS company TEXT;
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
        ALTER TABLE supervisor_chat_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
        UPDATE supervisor_chat_sessions
        SET id = COALESCE(id, session_id)
        WHERE id IS NULL;
        UPDATE supervisor_chat_sessions
        SET session_id = COALESCE(session_id, id)
        WHERE session_id IS NULL;
    """
    sql_messages = """
        CREATE TABLE IF NOT EXISTS supervisor_chat_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_sessions)
            cur.execute(sql_sessions_compat)
            cur.execute(sql_messages)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_supervisor_chat_messages_session_id ON supervisor_chat_messages(session_id);")
        conn.commit()


def create_supervisor_chat_session(
    *,
    title: str,
    symbol: Optional[str] = None,
    company: Optional[str] = None,
) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO supervisor_chat_sessions (id, session_id, title, symbol, company)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, title, symbol, company, created_at, updated_at;
                """,
                (session_id, session_id, title, symbol, company),
            )
            row = cur.fetchone()
        conn.commit()
    return {
        "id": row[0],
        "title": row[1],
        "symbol": row[2],
        "company": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


def list_supervisor_chat_sessions() -> List[Dict[str, Any]]:
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(id, session_id) AS id, title, symbol, company, created_at, updated_at
                FROM supervisor_chat_sessions
                ORDER BY updated_at DESC, created_at DESC;
                """
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "title": row[1],
            "symbol": row[2],
            "company": row[3],
            "created_at": row[4].isoformat(),
            "updated_at": row[5].isoformat(),
        }
        for row in rows
    ]


def get_supervisor_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(id, session_id) AS id, title, symbol, company, created_at, updated_at
                FROM supervisor_chat_sessions
                WHERE id = %s OR session_id = %s;
                """,
                (session_id, session_id),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "symbol": row[2],
        "company": row[3],
        "created_at": row[4].isoformat(),
        "updated_at": row[5].isoformat(),
    }


def update_supervisor_chat_session_context(
    *,
    session_id: str,
    symbol: Optional[str],
    company: Optional[str],
) -> None:
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE supervisor_chat_sessions
                SET symbol = COALESCE(%s, symbol),
                    company = COALESCE(%s, company),
                    updated_at = NOW()
                WHERE id = %s OR session_id = %s;
                """,
                (symbol, company, session_id, session_id),
            )
        conn.commit()


def add_supervisor_chat_message(*, session_id: str, role: str, content: str) -> Dict[str, Any]:
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO supervisor_chat_messages (session_id, role, content)
                VALUES (%s, %s, %s)
                RETURNING id, session_id, role, content, created_at;
                """,
                (session_id, role, content),
            )
            row = cur.fetchone()
            cur.execute(
                "UPDATE supervisor_chat_sessions SET updated_at = NOW() WHERE id = %s OR session_id = %s;",
                (session_id, session_id),
            )
        conn.commit()
    return {
        "id": row[0],
        "session_id": row[1],
        "role": row[2],
        "content": row[3],
        "created_at": row[4].isoformat(),
    }


def list_supervisor_chat_messages(session_id: str) -> List[Dict[str, Any]]:
    with psycopg.connect(_conn_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM supervisor_chat_messages
                WHERE session_id = %s
                ORDER BY id ASC;
                """,
                (session_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4].isoformat(),
        }
        for row in rows
    ]
