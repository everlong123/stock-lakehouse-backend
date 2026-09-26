"""Data ingestion and management endpoints for VN stocks, fundamental, news, and macro."""

from fastapi import APIRouter

from app.schemas.common import ok
from app.services.data_ingestion_service import get_ingestion_service

router = APIRouter(prefix="/data", tags=["data"])


# =====================
# VN Stock Data
# =====================

@router.get("/vn-stocks/{symbol}")
def get_vn_stock(symbol: str) -> dict:
    """Get VN stock OHLCV data."""
    service = get_ingestion_service()
    frame = service.vn_provider.get_historical_data(symbol.upper())
    return ok({
        "symbol": symbol.upper(),
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/vn-stocks/{symbol}/latest")
def get_vn_stock_latest(symbol: str) -> dict:
    """Get latest VN stock data."""
    service = get_ingestion_service()
    frame = service.vn_provider.get_latest_data(symbol.upper())
    return ok({
        "symbol": symbol.upper(),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/indices/{index_name}/components")
def get_index_components(index_name: str) -> dict:
    """Get stocks for a VN index (VN-Index, VN30, HNX, UPCOM)."""
    service = get_ingestion_service()
    components = service.vn_provider.get_index_components(index_name)
    return ok({
        "index": index_name,
        "components": components,
        "count": len(components),
    })


@router.post("/vn-stocks/ingest")
def ingest_vn_stocks(symbols: list[str] | None = None) -> dict:
    """Ingest VN stock data."""
    service = get_ingestion_service()
    results = service.ingest_vn_stocks(symbols)
    return ok({
        "results": results,
        "total": sum(r for r in results.values() if isinstance(r, int) and r > 0),
    })


# =====================
# Fundamental Data
# =====================

@router.get("/fundamental/{symbol}")
def get_fundamental(symbol: str) -> dict:
    """Get fundamental data for a symbol."""
    service = get_ingestion_service()
    data = service.fundamental_provider.get_financial_summary(symbol.upper())
    return ok({
        "symbol": symbol.upper(),
        "data": data,
    })


@router.post("/fundamental/ingest")
def ingest_fundamental(symbols: list[str]) -> dict:
    """Ingest fundamental data for symbols."""
    service = get_ingestion_service()
    results = service.ingest_fundamental(symbols)
    return ok({
        "results": results,
        "success_count": sum(1 for v in results.values() if v == "success"),
    })


# =====================
# News + Sentiment
# =====================

@router.get("/news/{symbol}")
def get_news(symbol: str, limit: int = 20) -> dict:
    """Get news for a symbol."""
    service = get_ingestion_service()
    frame = service.news_provider.get_news_for_symbol(symbol.upper(), limit)
    return ok({
        "symbol": symbol.upper(),
        "count": len(frame),
        "articles": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/news/market")
def get_market_news(limit: int = 30) -> dict:
    """Get general market news."""
    service = get_ingestion_service()
    frame = service.news_provider.get_market_news(limit)
    return ok({
        "count": len(frame),
        "articles": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/sentiment/{symbol}")
def get_sentiment(symbol: str, days: int = 7) -> dict:
    """Get sentiment summary for a symbol."""
    service = get_ingestion_service()
    summary = service.news_provider.get_sentiment_summary(symbol.upper(), days)
    return ok({
        "symbol": symbol.upper(),
        "summary": summary,
    })


@router.post("/news/ingest")
def ingest_news(symbols: list[str] | None = None, limit: int = 20) -> dict:
    """Ingest news data."""
    service = get_ingestion_service()
    frame = service.ingest_news(symbols, limit)
    return ok({
        "count": len(frame),
        "articles": frame.to_dict("records") if not frame.empty else [],
    })


# =====================
# Macro Data
# =====================

@router.get("/macro/exchange-rate")
def get_exchange_rate() -> dict:
    """Get USD/VND exchange rate."""
    service = get_ingestion_service()
    frame = service.macro_provider.get_exchange_rate()
    return ok({
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/macro/gold")
def get_gold_price() -> dict:
    """Get gold prices."""
    service = get_ingestion_service()
    frame = service.macro_provider.get_gold_price()
    return ok({
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/macro/oil")
def get_oil_price() -> dict:
    """Get oil prices (WTI, Brent)."""
    service = get_ingestion_service()
    frame = service.macro_provider.get_oil_price()
    return ok({
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/macro/interest-rates")
def get_interest_rates() -> dict:
    """Get interest rates."""
    service = get_ingestion_service()
    frame = service.macro_provider.get_interest_rates()
    return ok({
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/macro/cpi")
def get_cpi() -> dict:
    """Get CPI data."""
    service = get_ingestion_service()
    frame = service.macro_provider.get_cpi()
    return ok({
        "count": len(frame),
        "data": frame.to_dict("records") if not frame.empty else [],
    })


@router.get("/macro/summary")
def get_macro_summary() -> dict:
    """Get all macro indicators summary."""
    service = get_ingestion_service()
    summary = service.macro_provider.get_macro_summary()
    return ok({
        "data": summary,
    })


@router.post("/macro/ingest")
def ingest_macro() -> dict:
    """Ingest all macro data."""
    service = get_ingestion_service()
    results = service.ingest_all_macro()
    return ok({
        "results": results,
    })


# =====================
# Full Pipeline
# =====================

@router.post("/ingest/all")
def ingest_all() -> dict:
    """Run full data ingestion for all sources."""
    service = get_ingestion_service()
    summary = service.ingest_all()
    return ok({
        "data": summary,
    })
