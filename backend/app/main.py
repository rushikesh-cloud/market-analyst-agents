from __future__ import annotations

import os
from fastapi import FastAPI

from app.agents.news.web_search_agent import build_web_search_agent

app = FastAPI(title="Market Analyst Agent API")

# Lazy-init at import time for now; can be moved to startup event if desired.
web_search_agent = build_web_search_agent()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/web-search")
def run_web_search(payload: dict) -> dict:
    """
    Minimal endpoint to exercise the web search agent.
    Expects: {"query": "..."}
    """
    query = payload.get("query", "").strip()
    if not query:
        return {"error": "query is required"}

    result = web_search_agent.invoke({"input": query})
    return {"result": result}
