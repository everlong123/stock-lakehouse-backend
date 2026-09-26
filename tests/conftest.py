"""Shared test fixtures. All tests run offline on synthetic OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.constants import OHLCV_COLUMNS


def make_ohlcv(rows: int = 180, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=rows, tz="UTC")
    close = start_price * np.exp(np.cumsum(rng.normal(0.0004, 0.015, size=rows)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.integers(1_000_000, 5_000_000, size=rows).astype(float)
    return pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
            "source": "sample",
            "ingestion_time": pd.Timestamp.now(tz="UTC"),
        }
    )[OHLCV_COLUMNS]


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    return make_ohlcv()
