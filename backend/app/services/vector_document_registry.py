from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import psycopg


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _normalize_connection_string(connection_string: str) -> str:
    if connection_string.startswith("postgresql+psycopg://"):
        return connection_string.replace("postgresql+psycopg://", "postgresql://", 1)
    return connection_string


@dataclass
class IngestedDocumentRecord:
    collection_name: str
    source_path: str
    company: Optional[str]
    ticker: Optional[str]
    doc_type: Optional[str]
    year: Optional[str]
    chunks_stored: int


def list_ingested_documents(
    *,
    collection_name: Optional[str] = "fundamental_docs",
    connection_string: Optional[str] = None,
) -> List[IngestedDocumentRecord]:
    connection_string = _normalize_connection_string(connection_string or _env("PGVECTOR_CONNECTION_STRING"))
    base_sql = """
        SELECT
            c.name AS collection_name,
            COALESCE(e.cmetadata->>'source_path', '') AS source_path,
            NULLIF(e.cmetadata->>'company', '') AS company,
            NULLIF(e.cmetadata->>'ticker', '') AS ticker,
            NULLIF(e.cmetadata->>'doc_type', '') AS doc_type,
            NULLIF(e.cmetadata->>'year', '') AS year,
            COUNT(*)::int AS chunks_stored
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    """
    grouping_sql = """
        GROUP BY c.name, source_path, company, ticker, doc_type, year
        ORDER BY c.name, source_path
    """
    records: List[IngestedDocumentRecord] = []
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            if collection_name:
                cur.execute(f"{base_sql} WHERE c.name = %s {grouping_sql}", (collection_name,))
            else:
                cur.execute(f"{base_sql} {grouping_sql}")
            for row in cur.fetchall():
                source_path = row[1] or ""
                if not source_path:
                    continue
                records.append(
                    IngestedDocumentRecord(
                        collection_name=row[0],
                        source_path=source_path,
                        company=row[2],
                        ticker=row[3],
                        doc_type=row[4],
                        year=row[5],
                        chunks_stored=int(row[6] or 0),
                    )
                )
    return records


def delete_ingested_document(
    *,
    source_path: str,
    collection_name: str = "fundamental_docs",
    connection_string: Optional[str] = None,
) -> int:
    connection_string = _normalize_connection_string(connection_string or _env("PGVECTOR_CONNECTION_STRING"))
    sql = """
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE
            e.collection_id = c.uuid
            AND c.name = %s
            AND COALESCE(e.cmetadata->>'source_path', '') = %s
        RETURNING e.uuid
    """
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (collection_name, source_path))
            deleted_rows = cur.fetchall()
        conn.commit()
    return len(deleted_rows)
