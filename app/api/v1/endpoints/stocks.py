"""Stock market data endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_market_service
from app.core.config import settings
from app.schemas.common import ok
from app.services.market_service import MarketService

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("/symbols")
def get_symbols() -> dict:
    symbols = [s.strip().upper() for s in settings.crawl_symbols.split(",") if s.strip()]
    return ok({"symbols": symbols})


@router.get("/{symbol}")
def get_stock(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = Query(default="1d"),
    service: MarketService = Depends(get_market_service),
) -> dict:
    frame = service.get_history(symbol.upper(), start=start, end=end, interval=interval)
    latest = service.get_latest(symbol.upper(), interval=interval)
    return ok(
        {
            "symbol": symbol.upper(),
            "interval": interval,
            "latest": latest,
            "count": int(len(frame)),
            "rows": service.to_records(frame),
            "last_updated": latest["timestamp"],
            "data_source": latest.get("source_layer"),
        }
    )


@router.get("/{symbol}/latest")
def get_latest(symbol: str, interval: str = "1d", service: MarketService = Depends(get_market_service)) -> dict:
    return ok(service.get_latest(symbol.upper(), interval=interval))


@router.get("/{symbol}/csv")
def download_csv(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = "1d",
    service: MarketService = Depends(get_market_service),
) -> StreamingResponse:
    frame = service.get_history(symbol.upper(), start=start, end=end, interval=interval)
    csv_data = frame[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].to_csv(index=False)
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={symbol.upper()}_{interval}.csv"},
    )
