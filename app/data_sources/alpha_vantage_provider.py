"""Alpha Vantage provider for real-time stock data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS, SUPPORTED_INTERVALS
from app.core.exceptions import DataSourceError, DataValidationError
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider

logger = get_logger(__name__)

_ALPHA_INTERVAL_MAP = {
    "1d": "Daily",
    "1h": "60min",
    "15m": "15min",
    "5m": "5min",
}


class AlphaVantageProvider(StockDataProvider):
    """Fetch OHLCV data via Alpha Vantage API."""

    source_name = "alpha_vantage"

    def __init__(self):
        self.api_key = settings.alpha_vantage_api_key
        if not self.api_key:
            raise DataSourceError("Alpha Vantage API key not configured. Set ALPHA_VANTAGE_API_KEY in .env")
        self.base_url = "https://www.alphavantage.co/query"

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
        
        # Alpha Vantage only supports Daily (1d) and intraday (60min, 15min, 5min)
        alpha_function = "TIME_SERIES_DAILY_ADJUSTED" if interval == "1d" else f"TIME_SERIES_INTRADAY&interval={interval}"
        
        end = end or datetime.now(timezone.utc)
        start = start or datetime.now(timezone.utc) - timedelta(days=settings.default_lookback_days)
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED" if interval == "1d" else f"TIME_SERIES_INTRADAY&interval={interval}",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise DataSourceError(f"Alpha Vantage request failed: {exc}") from exc
        
        # Parse response
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "TimeSeries" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            error_msg = data.get("Error Message") or data.get("Note") or data.get("Information", "Unknown error")
            raise DataSourceError(f"Alpha Vantage error: {error_msg}")
        
        time_series = data[time_series_key]
        records = []
        
        for date_str, values in time_series.items():
            dt = datetime.fromisoformat(date_str)
            if dt >= end:
                continue
            if dt <= start:
                break
            
            records.append({
                "timestamp": dt.replace(tzinfo=timezone.utc),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "adj_close": float(values.get("5. adjusted close", values.get("4. close", 0))),
                "volume": float(values.get("6. volume", 0)),
            })
        
        if not records:
            raise DataSourceError(f"No data returned from Alpha Vantage for {symbol}")
        
        frame = pd.DataFrame(records)
        frame["symbol"] = symbol
        frame["source"] = self.source_name
        frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        
        logger.info("Fetched %s Alpha Vantage rows for %s", len(frame), symbol)
        return frame

    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=14)
        frame = self.get_historical_data(symbol=symbol, start=start, end=end, interval=interval)
        return frame.tail(1).reset_index(drop=True)
