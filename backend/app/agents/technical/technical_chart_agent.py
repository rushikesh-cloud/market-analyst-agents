from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage

# Force headless backend to avoid Tkinter main-loop errors in API/server contexts.
matplotlib.use("Agg")
import mplfinance as mpf


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass
class TechnicalAnalysisResult:
    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, float]


def _fetch_price_data(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")
    # yfinance often returns a MultiIndex with the ticker as level 1
    if isinstance(data.columns, pd.MultiIndex):
        tickers = list(data.columns.get_level_values(1).unique())
        if symbol in tickers:
            data = data.xs(symbol, axis=1, level=1)
        else:
            data = data.xs(tickers[0], axis=1, level=1)
    data = data.rename(columns=str.title)
    return data


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.macd(close="Close", append=True)
    df.ta.rsi(close="Close", length=14, append=True)
    return df


def _plot_chart(df: pd.DataFrame, symbol: str, out_path: Path) -> None:
    macd = df[["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]]
    rsi = df["RSI_14"]

    apds = [
        mpf.make_addplot(macd["MACD_12_26_9"], panel=1, color="fuchsia", ylabel="MACD"),
        mpf.make_addplot(macd["MACDs_12_26_9"], panel=1, color="b"),
        mpf.make_addplot(macd["MACDh_12_26_9"], type="bar", panel=1, color="dimgray"),
        mpf.make_addplot(rsi, panel=2, color="g", ylabel="RSI"),
    ]

    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle="-", gridcolor="lightgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=apds,
        volume=True,
        panel_ratios=(6, 2, 2),
        title=f"{symbol} | Daily Candles + MACD + RSI",
        savefig=dict(fname=str(out_path), dpi=150, bbox_inches="tight"),
    )


def _image_to_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _vision_analyze(image_path: Path, symbol: str) -> str:
    llm = AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.2,
    )

    data_url = _image_to_data_url(image_path)

    prompt = (
        "You are a technical analysis expert. Analyze the candlestick chart with MACD and RSI. "
        "Provide a concise summary of trend, momentum, notable crossovers, RSI overbought/oversold, "
        "and any potential near-term signals."
    )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"Symbol: {symbol}\n{prompt}"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    response = llm.invoke([msg])
    return response.content


def analyze_stock_technical(symbol: str, period: str = "3mo", interval: str = "1d") -> TechnicalAnalysisResult:
    df = _fetch_price_data(symbol, period=period, interval=interval)
    df = _add_indicators(df)

    out_path = Path("data/processed") / f"{symbol}_technical.png"
    _plot_chart(df, symbol, out_path)

    summary = _vision_analyze(out_path, symbol)

    latest = df.iloc[-1]
    latest_values = {
        "close": float(latest["Close"]),
        "rsi_14": float(latest.get("RSI_14", float("nan"))),
        "macd": float(latest.get("MACD_12_26_9", float("nan"))),
        "macd_signal": float(latest.get("MACDs_12_26_9", float("nan"))),
        "macd_hist": float(latest.get("MACDh_12_26_9", float("nan"))),
    }

    return TechnicalAnalysisResult(
        symbol=symbol,
        image_path=str(out_path),
        summary=summary,
        latest_values=latest_values,
    )
