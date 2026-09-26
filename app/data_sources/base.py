"""Abstract market data provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class StockDataProvider(ABC):
    """Contract for historical and latest OHLCV retrieval."""

    source_name: str = "abstract"

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return historical OHLCV for a symbol."""

    @abstractmethod
    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """Return the most recent bar(s) for a symbol."""

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Return True if the symbol is supported by this provider."""
