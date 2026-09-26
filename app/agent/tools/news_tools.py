"""News and sentiment tools for AI agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider


class NewsInput(BaseModel):
    symbol: str = Field(description="Ticker symbol, e.g. VCB, AAPL")
    limit: int = Field(default=10, ge=1, le=50, description="Số lượng bài viết")


class SentimentSummaryInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    days: int = Field(default=7, ge=1, le=90, description="Số ngày phân tích")


class SentimentTimeseriesInput(BaseModel):
    symbol: str = Field(description="Ticker symbol")
    days: int = Field(default=30, ge=1, le=90, description="Số ngày")


class MarketNewsInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=30, description="Số lượng tin thị trường")


def get_news_for_symbol(symbol: str, limit: int = 10) -> dict[str, Any]:
    """Lấy tin tức có phân tích sentiment cho một mã."""
    provider = EnhancedNewsSentimentProvider()
    df = provider.get_news_for_symbol(symbol.upper(), limit=limit)

    if df.empty:
        return {
            "symbol": symbol.upper(),
            "articles": [],
            "count": 0,
            "message": "Không tìm thấy tin tức cho mã này.",
        }

    articles = []
    for _, row in df.iterrows():
        articles.append({
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "published_date": str(row.get("published_date", "")),
            "source": row.get("source", ""),
            "sentiment": float(row.get("sentiment", 0)),
            "sentiment_label": row.get("sentiment_label", "neutral"),
            "sentiment_confidence": float(row.get("sentiment_confidence", 0)),
        })

    return {
        "symbol": symbol.upper(),
        "articles": articles,
        "count": len(articles),
        "avg_sentiment": float(df["sentiment"].mean()) if not df.empty else 0,
        "positive_count": int((df["sentiment"] > 0.2).sum()),
        "negative_count": int((df["sentiment"] < -0.2).sum()),
    }


def get_sentiment_summary(symbol: str, days: int = 7) -> dict[str, Any]:
    """Lấy tóm tắt sentiment cho một mã trong N ngày."""
    provider = EnhancedNewsSentimentProvider()
    return provider.get_sentiment_summary(symbol.upper(), days=days)


def get_sentiment_timeseries(symbol: str, days: int = 30) -> dict[str, Any]:
    """Lấy dữ liệu sentiment theo thời gian."""
    provider = EnhancedNewsSentimentProvider()
    df = provider.get_sentiment_timeseries(symbol.upper(), days=days)

    if df.empty:
        return {
            "symbol": symbol.upper(),
            "data": [],
            "count": 0,
        }

    return {
        "symbol": symbol.upper(),
        "data": df.to_dict("records"),
        "count": len(df),
        "trend": "positive" if df["sentiment"].mean() > 0.2 else "negative" if df["sentiment"].mean() < -0.2 else "neutral",
    }


def get_market_news(limit: int = 10) -> dict[str, Any]:
    """Lấy tin tức thị trường chung."""
    provider = EnhancedNewsSentimentProvider()
    df = provider.get_market_news(limit=limit)

    if df.empty:
        return {
            "articles": [],
            "count": 0,
        }

    articles = []
    for _, row in df.iterrows():
        articles.append({
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "source": row.get("source", ""),
            "sentiment": float(row.get("sentiment", 0)),
        })

    avg_sentiment = df["sentiment"].mean() if not df.empty else 0

    return {
        "articles": articles,
        "count": len(articles),
        "avg_sentiment": float(avg_sentiment),
        "sentiment_label": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral",
    }
