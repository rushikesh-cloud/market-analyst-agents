from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

from langchain_openai import AzureChatOpenAI


class GuardrailViolation(ValueError):
    """Raised when a request is outside market/stock analysis scope."""


logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_guardrail_llm: AzureChatOpenAI | None = None


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _get_guardrail_llm() -> AzureChatOpenAI:
    global _guardrail_llm
    if _guardrail_llm is None:
        deployment = os.getenv("AZURE_OPENAI_GUARDRAIL_DEPLOYMENT") or _env("AZURE_OPENAI_DEPLOYMENT")
        _guardrail_llm = AzureChatOpenAI(
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_KEY"),
            azure_deployment=deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.0,
        )
    return _guardrail_llm


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty response", cleaned, 0)

    if "```" in cleaned:
        chunks = cleaned.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    return json.loads(cleaned)


def _llm_market_scope_decision(query: str, *, field_name: str) -> Dict[str, Any]:
    llm = _get_guardrail_llm()
    system_prompt = (
        "You are a strict API guardrail classifier. "
        "Allow only market, stock, trading, investment, company fundamentals, or macro-finance related queries. "
        "Disallow unrelated domains (e.g., cooking, travel, poetry, coding help unrelated to markets, general chit-chat). "
        "Return ONLY JSON with keys: "
        '{"allowed": <true|false>, "reason": "<short reason>"}'
    )
    user_prompt = (
        f"Field: {field_name}\n"
        f"Query: {query}\n"
        "Decide if this is in-scope for market/stock analysis APIs."
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        content = "\n".join(parts)
    parsed = _parse_json_object(str(content))
    return parsed


def ensure_market_query(query: str, *, field_name: str = "query") -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise GuardrailViolation(f"{field_name} must not be empty.")
    try:
        decision = _llm_market_scope_decision(cleaned, field_name=field_name)
    except Exception:
        logger.exception("Guardrail LLM classification failed for field=%s", field_name)
        raise GuardrailViolation("Guardrail validation failed. Please retry with a market-related query.")

    allowed = bool(decision.get("allowed", False))
    reason = str(decision.get("reason", "") or "").strip()
    if not allowed:
        suffix = f" Reason: {reason}" if reason else ""
        raise GuardrailViolation(
            f"{field_name} is out of scope. Only market and stock related queries are supported.{suffix}"
        )
    return cleaned


def ensure_stock_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise GuardrailViolation("symbol must not be empty.")
    if not _SYMBOL_RE.match(cleaned):
        raise GuardrailViolation(
            "symbol is invalid. Use a valid stock ticker format like AAPL, MSFT, RELIANCE, BRK.B."
        )
    return cleaned
