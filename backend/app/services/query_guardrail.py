from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str


@lru_cache(maxsize=1)
def _guardrail_llm() -> AzureChatOpenAI:
    # Use a cheaper/smaller deployment when provided.
    deployment = os.getenv("AZURE_OPENAI_GUARDRAIL_DEPLOYMENT") or _env("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_GUARDRAIL_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    return AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0,
        max_tokens=120,
    )


def _parse_guardrail_response(text: str) -> GuardrailDecision:
    content = (text or "").strip()
    if not content:
        return GuardrailDecision(allowed=False, reason="Empty guardrail response")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return GuardrailDecision(allowed=False, reason=f"Invalid guardrail response: {content[:180]}")
        payload = json.loads(content[start : end + 1])

    allowed = bool(payload.get("allowed", False))
    reason = str(payload.get("reason", "") or "No reason provided.")
    return GuardrailDecision(allowed=allowed, reason=reason)


def validate_market_query(
    *,
    query: str,
    company: Optional[str] = None,
    symbol: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> GuardrailDecision:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return GuardrailDecision(allowed=False, reason="Query is empty.")

    context = {
        "agent_name": agent_name or "",
        "company": (company or "").strip(),
        "symbol": (symbol or "").strip().upper(),
        "query": cleaned_query,
    }

    system_prompt = (
        "You are a strict request classifier for a market analysis system.\n"
        "Allow ONLY queries related to company markets/investing/trading/financial analysis.\n"
        "Allowed examples: stock price/technicals, earnings, fundamentals, valuation, risks, catalysts, news that can impact the stock.\n"
        "Reject unrelated requests: coding help, travel, recipes, health, general trivia, or non-market tasks.\n"
        "Return ONLY JSON with schema:\n"
        '{"allowed": true|false, "reason": "short reason"}'
    )

    human_prompt = (
        "Classify this request.\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )

    resp = _guardrail_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    return _parse_guardrail_response(str(resp.content))

