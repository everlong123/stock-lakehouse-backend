"""API endpoints for market indices, order book, and enhanced data."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/indices")
async def get_all_indices():
    """Get all market indices (VN-Index, HNX, UPCOM, VN30, etc.)."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    return provider.get_all_indices()


@router.get("/indices/{symbol}")
async def get_index(
    symbol: str,
    lookback_days: int = Query(30, ge=1, le=365),
):
    """Get historical data for a specific index."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    return provider.get_index_data(symbol.upper())


@router.get("/indices/{symbol}/summary")
async def get_index_summary(symbol: str):
    """Get current summary for an index."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    return provider.get_index_summary(symbol.upper())


@router.get("/indices/{symbol}/breadth")
async def get_market_breadth(symbol: str = "VNINDEX"):
    """Get market breadth data."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    return provider.get_marketBreadth()


@router.get("/indices/sectors")
async def get_sector_performance():
    """Get sector performance data."""
    from app.data_sources.market_index_provider import MarketIndexProvider

    provider = MarketIndexProvider()
    df = provider.get_sector_performance()
    return df.to_dict("records") if not df.empty else []


# ─── Order Book Endpoints ────────────────────────────────────────────────────────


@router.get("/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    depth: int = Query(10, ge=1, le=50),
):
    """Get current order book for a symbol."""
    from app.data_sources.orderbook_provider import OrderBookProvider

    provider = OrderBookProvider()
    orderbook = provider.get_order_book(symbol.upper(), depth=depth)

    if orderbook is None:
        return {"error": "Order book unavailable for this symbol"}

    return {
        "symbol": orderbook.symbol,
        "timestamp": orderbook.timestamp.isoformat(),
        "best_bid": orderbook.best_bid,
        "best_ask": orderbook.best_ask,
        "spread": orderbook.spread,
        "mid_price": orderbook.mid_price,
        "bids": [
            {"price": b.price, "volume": b.volume, "orders": b.orders}
            for b in orderbook.bids
        ],
        "asks": [
            {"price": a.price, "volume": a.volume, "orders": a.orders}
            for a in orderbook.asks
        ],
    }


@router.get("/trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
):
    """Get recent trades for a symbol."""
    from app.data_sources.orderbook_provider import OrderBookProvider

    provider = OrderBookProvider()
    df = provider.get_recent_trades(symbol.upper(), limit=limit)
    return df.to_dict("records") if not df.empty else []


@router.get("/trades/{symbol}/summary")
async def get_trade_summary(symbol: str):
    """Get trade summary statistics."""
    from app.data_sources.orderbook_provider import OrderBookProvider

    provider = OrderBookProvider()
    return provider.get_trade_summary(symbol.upper())


# ─── Enhanced Fundamental Endpoints ─────────────────────────────────────────────


@router.get("/fundamental/{symbol}")
async def get_fundamental_metrics(symbol: str):
    """Get comprehensive fundamental metrics for a symbol."""
    from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider

    provider = EnhancedFundamentalProvider()
    metrics = provider.get_comprehensive_metrics(symbol.upper())
    return provider.metrics_to_dict(metrics)


@router.get("/fundamental/{symbol}/income-statement")
async def get_income_statement(
    symbol: str,
    years: int = Query(4, ge=1, le=10),
):
    """Get income statement data."""
    from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider

    provider = EnhancedFundamentalProvider()
    df = provider.get_income_statement(symbol.upper(), years=years)
    return df.to_dict("records") if not df.empty else []


@router.get("/fundamental/{symbol}/balance-sheet")
async def get_balance_sheet(
    symbol: str,
    years: int = Query(4, ge=1, le=10),
):
    """Get balance sheet data."""
    from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider

    provider = EnhancedFundamentalProvider()
    df = provider.get_balance_sheet(symbol.upper(), years=years)
    return df.to_dict("records") if not df.empty else []


@router.get("/fundamental/{symbol}/cash-flow")
async def get_cash_flow(
    symbol: str,
    years: int = Query(4, ge=1, le=10),
):
    """Get cash flow data."""
    from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider

    provider = EnhancedFundamentalProvider()
    df = provider.get_cash_flow(symbol.upper(), years=years)
    return df.to_dict("records") if not df.empty else []


# ─── Enhanced News/Sentiment Endpoints ──────────────────────────────────────────


@router.get("/news/{symbol}")
async def get_news_with_sentiment(
    symbol: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Get news articles with NLP sentiment analysis."""
    from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider

    provider = EnhancedNewsSentimentProvider()
    df = provider.get_news_for_symbol(symbol.upper(), limit=limit)
    return df.to_dict("records") if not df.empty else []


@router.get("/news/market")
async def get_market_news(
    limit: int = Query(30, ge=1, le=100),
):
    """Get general market news."""
    from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider

    provider = EnhancedNewsSentimentProvider()
    df = provider.get_market_news(limit=limit)
    return df.to_dict("records") if not df.empty else []


@router.get("/news/{symbol}/sentiment-summary")
async def get_sentiment_summary(
    symbol: str,
    days: int = Query(7, ge=1, le=90),
):
    """Get aggregated sentiment summary for a symbol."""
    from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider

    provider = EnhancedNewsSentimentProvider()
    return provider.get_sentiment_summary(symbol.upper(), days=days)


@router.get("/news/{symbol}/sentiment-timeseries")
async def get_sentiment_timeseries(
    symbol: str,
    days: int = Query(30, ge=1, le=90),
):
    """Get sentiment data over time."""
    from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider

    provider = EnhancedNewsSentimentProvider()
    df = provider.get_sentiment_timeseries(symbol.upper(), days=days)
    return df.to_dict("records") if not df.empty else []
