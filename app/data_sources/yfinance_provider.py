"""Yahoo Finance provider. Used only when DATA_SOURCE=yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS, SUPPORTED_INTERVALS
from app.core.exceptions import DataSourceError, DataValidationError
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider

logger = get_logger(__name__)

_YF_INTERVAL_MAP = {
    "1d": "1d",
    "1h": "1h",
    "15m": "15m",
    "5m": "5m",
}


class YFinanceProvider(StockDataProvider):
    """Fetch near-real-time OHLCV via polling yfinance. Not tick-level."""

    source_name = "yfinance"

    def validate_symbol(self, symbol: str) -> bool:
        try:
            frame = self.get_latest_data(symbol)
            return not frame.empty
        except Exception:
            return False

    def get_historical_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        if interval not in SUPPORTED_INTERVALS:
            raise DataValidationError(f"Unsupported interval: {interval}")
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataSourceError("yfinance is not installed.") from exc

        ticker = yf.Ticker(symbol)
        start = start or datetime.now(timezone.utc) - timedelta(days=settings.default_lookback_days)
        end = end or datetime.now(timezone.utc)
        try:
            raw = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=_YF_INTERVAL_MAP[interval],
                auto_adjust=False,
                timeout=settings.yfinance_timeout,
            )
        except Exception as exc:
            raise DataSourceError(f"yfinance request failed for {symbol}: {exc}") from exc

        if raw is None or raw.empty:
            raise DataSourceError(
                f"yfinance returned no rows for {symbol}. "
                "The platform can still run offline with DATA_SOURCE=sample."
            )

        frame = raw.reset_index()
        time_col = "Datetime" if "Datetime" in frame.columns else "Date"
        frame = frame.rename(
            columns={
                time_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame["close"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = symbol
        frame["source"] = self.source_name
        frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        logger.info("Fetched %s yfinance rows for %s interval=%s", len(frame), symbol, interval)
        return frame

    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=14)
        frame = self.get_historical_data(symbol=symbol, start=start, end=end, interval=interval)
        return frame.tail(1).reset_index(drop=True)
