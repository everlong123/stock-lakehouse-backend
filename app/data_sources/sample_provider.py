"""Offline sample OHLCV provider used for demos without internet."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS, SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS
from app.core.exceptions import DataSourceError, DataValidationError
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider

logger = get_logger(__name__)


class SampleDataProvider(StockDataProvider):
    """Read synthetic OHLCV from local CSV/Parquet files."""

    source_name = "sample"

    def __init__(self, sample_dir: Path | None = None) -> None:
        self.sample_dir = sample_dir or (settings.data_root_path / "sample")

    def _file_path(self, symbol: str, interval: str) -> Path:
        symbol = symbol.upper()
        parquet_path = self.sample_dir / f"{symbol}_{interval}.parquet"
        csv_path = self.sample_dir / f"{symbol}_{interval}.csv"
        if parquet_path.exists():
            return parquet_path
        if csv_path.exists():
            return csv_path
        raise DataSourceError(
            f"Sample data not found for {symbol} ({interval}). "
            "Run python scripts/generate_sample_data.py first."
        )

    def validate_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in SUPPORTED_SYMBOLS:
            return False
        try:
            self._file_path(symbol, "1d")
            return True
        except DataSourceError:
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
        path = self._file_path(symbol, interval)
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
        if frame.empty:
            raise DataSourceError(f"Sample dataset for {symbol} is empty.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            frame = frame[frame["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            frame = frame[frame["timestamp"] <= end_ts]
        frame["symbol"] = symbol
        frame["source"] = self.source_name
        if "ingestion_time" not in frame.columns:
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
        if missing:
            raise DataSourceError(f"Sample file missing columns: {missing}")
        frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %s sample rows for %s interval=%s", len(frame), symbol, interval)
        return frame

    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        frame = self.get_historical_data(symbol=symbol, interval=interval)
        return frame.tail(1).reset_index(drop=True)
