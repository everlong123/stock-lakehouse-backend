"""Ingest market data into the Bronze layer."""

from __future__ import annotations

from datetime import datetime

from app.core.logging_config import get_logger
from app.data_sources.factory import get_data_provider
from app.lakehouse.bronze import BronzeLayer

logger = get_logger(__name__)


def ingest_symbol(
    symbol: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    source_name: str | None = None,
) -> dict:
    """Fetch OHLCV and append it to Bronze with lineage metadata."""
    provider = get_data_provider(source_name)
    frame = provider.get_historical_data(symbol=symbol, start=start, end=end, interval=interval)
    bronze = BronzeLayer()
    metadata = bronze.append(
        frame,
        lineage={
            "provider": provider.source_name,
            "symbol": symbol.upper(),
            "interval": interval,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
    )
    logger.info("Ingestion complete for %s: %s rows", symbol, metadata["records_received"])
    return metadata
