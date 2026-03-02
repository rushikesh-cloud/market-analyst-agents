from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

from app.agents.fundamental.fundamental_agent import analyze_fundamentals
from app.agents.news.web_search_agent import build_web_search_agent, invoke_market_web_search, parse_web_search_result
from app.agents.technical.technical_chart_agent import analyze_stock_technical
from app.guardrails import ensure_market_query, ensure_stock_symbol

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _extract_final_text(payload: Any) -> str:
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
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if txt:
                            parts.append(str(txt))
                    elif item is not None:
                        parts.append(str(item))
                if parts:
                    return "\n".join(parts)
            if content:
                return str(content)
        output = payload.get("output")
        if isinstance(output, str):
            return output
        if output is not None:
            return str(output)
    return str(payload)


def _to_agent_messages_input(user_text: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": user_text}]}


def _build_supervisor_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.1,
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    return json.loads(text)


def _build_supervisor_agent(
    *,
    symbol: str,
    company: str,
    technical_period: str,
    technical_interval: str,
    collection_name: str,
    top_k: int,
):
    llm = _build_supervisor_llm()
    logger.debug("Building supervisor agent: symbol=%s company=%s collection=%s top_k=%s", symbol, company, collection_name, top_k)

    tool_outputs: Dict[str, Any] = {}

    @tool("technical_subagent")
    def technical_subagent(period: str = technical_period, interval: str = technical_interval) -> str:
        """Run technical analysis subagent for the target stock symbol."""
        logger.info("Supervisor -> technical_subagent")
        result = analyze_stock_technical(symbol, period=period, interval=interval)
        payload = {
            "symbol": result.symbol,
            "image_path": result.image_path,
            "summary": result.summary,
            "latest_values": result.latest_values,
        }
        tool_outputs["technical"] = payload
        return json.dumps(payload)

    @tool("fundamental_subagent")
    def fundamental_subagent(question: str = "") -> str:
        """Run fundamental analysis subagent for the target company."""
        logger.info("Supervisor -> fundamental_subagent")
        cleaned_question = (question or "").strip() or None
        if cleaned_question:
            cleaned_question = ensure_market_query(cleaned_question, field_name="fundamental_question")
        result = analyze_fundamentals(
            company=company,
            question=cleaned_question,
            mode="auto",
            collection_name=collection_name,
            top_k=top_k,
        )
        answer_text = _extract_final_text(result.answer)
        payload = {
            "mode": result.mode,
            "company": result.company,
            "answer": answer_text,
            "sources": result.sources,
        }
        tool_outputs["fundamental"] = payload
        return json.dumps(payload)

    @tool("news_subagent")
    def news_subagent(query: str = "") -> str:
        """Run web-news subagent for company and symbol context."""
        logger.info("Supervisor -> news_subagent")
        cleaned_query = (query or "").strip() or f"{company} {symbol} latest company news catalysts risks"
        agent = build_web_search_agent()
        result = invoke_market_web_search(agent, cleaned_query)
        parsed = parse_web_search_result(result, cleaned_query)
        payload = {
            "query": parsed["query"],
            "answer": parsed["answer"],
            "sources": parsed["sources"],
        }
        tool_outputs["news"] = payload
        return json.dumps(payload)

    system_prompt = (
        "You are the supervisor investment analyst. "
        "You MUST call these three tools exactly once each before finalizing: "
        "technical_subagent, fundamental_subagent, news_subagent. "
        "After tool calls, produce ONLY valid JSON with schema: "
        "{"
        '"investment_rating_6m": <integer 1-10>, '
        '"stance": "<Bullish|Neutral|Bearish>", '
        '"technical_section": "<short paragraph>", '
        '"fundamental_section": "<short paragraph>", '
        '"news_section": "<short paragraph>", '
        '"risks": ["<risk 1>", "<risk 2>", "<risk 3>"], '
        '"final_thesis": "<concise integrated thesis for next 6 months>"'
        "}. "
        "Do not include markdown or extra keys. "
        "Do not invent facts not present in tool outputs."
    )
    agent = create_agent(llm, [technical_subagent, fundamental_subagent, news_subagent], system_prompt=system_prompt)
    return agent, tool_outputs


def _default_synthesis_fallback(message: str) -> Dict[str, Any]:
    return {
        "investment_rating_6m": None,
        "stance": "Neutral",
        "technical_section": "",
        "fundamental_section": "",
        "news_section": "",
        "risks": [message],
        "final_thesis": message,
    }


def _normalize_synthesis(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    if "investment_rating_6m" in synthesis:
        try:
            rating = synthesis["investment_rating_6m"]
            if rating is not None:
                synthesis["investment_rating_6m"] = int(rating)
        except (TypeError, ValueError):
            synthesis["investment_rating_6m"] = None
    else:
        synthesis["investment_rating_6m"] = None

    for key in ("stance", "technical_section", "fundamental_section", "news_section", "final_thesis"):
        synthesis[key] = str(synthesis.get(key, "") or "")

    risks = synthesis.get("risks", [])
    if not isinstance(risks, list):
        risks = [str(risks)]
    synthesis["risks"] = [str(item) for item in risks]
    return synthesis


def _build_fallback_sections(
    *,
    technical: Dict[str, Any],
    fundamental: Dict[str, Any],
    news: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "investment_rating_6m": None,
        "stance": "Neutral",
        "technical_section": str(technical.get("summary", "") or ""),
        "fundamental_section": str(fundamental.get("answer", "") or ""),
        "news_section": str(news.get("answer", "") or ""),
        "risks": ["Supervisor synthesis failed; raw subagent outputs returned."],
        "final_thesis": "Unable to produce structured synthesis, review sections above.",
    }


def _invoke_supervisor_synthesis(
    *,
    symbol: str,
    company: str,
    fundamental_question: Optional[str],
    news_query: Optional[str],
    agent: Any,
) -> Dict[str, Any]:
    prompt = (
        f"Symbol: {symbol}\n"
        f"Company: {company}\n"
        f"Fundamental focus question: {(fundamental_question or '').strip() or 'General financial strength and risks'}\n"
        f"News focus query: {(news_query or '').strip() or f'{company} {symbol} latest company news catalysts risks'}\n"
        "Run all required tools, then provide the final JSON."
    )
    logger.info("Running supervisor synthesis")
    response = agent.invoke(_to_agent_messages_input(prompt))
    text = _extract_final_text(response)
    try:
        return _normalize_synthesis(_parse_json_object(text))
    except json.JSONDecodeError:
        logger.error("Supervisor synthesis returned invalid JSON")
        return _default_synthesis_fallback(text or "Synthesis response was not valid JSON.")


@dataclass
class SupervisorAnalysisResult:
    symbol: str
    company: str
    technical: Dict[str, Any]
    fundamental: Dict[str, Any]
    news: Dict[str, Any]
    synthesis: Dict[str, Any]


def analyze_market_supervised(
    *,
    symbol: str,
    company: str,
    fundamental_question: Optional[str] = None,
    news_query: Optional[str] = None,
    technical_period: str = "3mo",
    technical_interval: str = "1d",
    collection_name: str = "fundamental_docs",
    top_k: int = 8,
) -> SupervisorAnalysisResult:
    symbol = ensure_stock_symbol(symbol)
    company = company.strip().upper()
    logger.info("Starting supervised analysis: symbol=%s company=%s", symbol, company)
    if fundamental_question:
        fundamental_question = ensure_market_query(fundamental_question, field_name="fundamental_question")
    if news_query:
        news_query = ensure_market_query(news_query, field_name="news_query")
    agent, tool_outputs = _build_supervisor_agent(
        symbol=symbol,
        company=company,
        technical_period=technical_period,
        technical_interval=technical_interval,
        collection_name=collection_name,
        top_k=top_k,
    )

    try:
        synthesis = _invoke_supervisor_synthesis(
            symbol=symbol,
            company=company,
            fundamental_question=fundamental_question,
            news_query=news_query,
            agent=agent,
        )

        technical_payload = tool_outputs.get("technical", {})
        fundamental_payload = tool_outputs.get("fundamental", {})
        news_payload = tool_outputs.get("news", {})

        # If the supervisor missed a tool call, fill gaps deterministically.
        if not technical_payload:
            technical_result = analyze_stock_technical(symbol, period=technical_period, interval=technical_interval)
            technical_payload = {
                "symbol": technical_result.symbol,
                "image_path": technical_result.image_path,
                "summary": technical_result.summary,
                "latest_values": technical_result.latest_values,
            }
        if not fundamental_payload:
            fundamental_result = analyze_fundamentals(
                company=company,
                question=fundamental_question.strip() if fundamental_question else None,
                mode="auto",
                collection_name=collection_name,
                top_k=top_k,
            )
            fundamental_payload = {
                "mode": fundamental_result.mode,
                "company": fundamental_result.company,
                "answer": _extract_final_text(fundamental_result.answer),
                "sources": fundamental_result.sources,
            }
        if not news_payload:
            query = (news_query or f"{company} {symbol} latest company news catalysts risks").strip()
            web_agent = build_web_search_agent()
            news_result = invoke_market_web_search(web_agent, query)
            parsed_news = parse_web_search_result(news_result, query)
            news_payload = {
                "query": parsed_news["query"],
                "answer": parsed_news["answer"],
                "sources": parsed_news["sources"],
            }

        if not synthesis.get("technical_section") and not synthesis.get("fundamental_section") and not synthesis.get("news_section"):
            synthesis = _build_fallback_sections(
                technical=technical_payload,
                fundamental=fundamental_payload,
                news=news_payload,
            )
        logger.info("Completed supervised analysis: symbol=%s company=%s", symbol, company)
        return SupervisorAnalysisResult(
            symbol=symbol,
            company=company,
            technical=technical_payload,
            fundamental=fundamental_payload,
            news=news_payload,
            synthesis=synthesis,
        )
    except Exception:
        logger.exception("Supervisor analysis failed: symbol=%s company=%s", symbol, company)
        raise
