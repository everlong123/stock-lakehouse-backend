"""Generate offline OHLCV sample datasets for thesis demos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS, SUPPORTED_SYMBOLS
from app.core.logging_config import get_logger
from app.core.seeding import set_global_seed

logger = get_logger(__name__)

START_PRICES = {
    "AAPL": 150.0,
    "MSFT": 280.0,
    "GOOGL": 120.0,
    "TSLA": 180.0,
    "NVDA": 90.0,
    "AMZN": 130.0,
    "META": 220.0,
}


def simulate_ohlcv(symbol: str, periods: int, freq: str, seed_offset: int) -> pd.DataFrame:
    """Geometric Brownian Motion with realistic OHLC constraints."""
    rng = np.random.default_rng(settings.random_seed + seed_offset)
    if freq == "B":
        index = pd.bdate_range("2021-01-04", periods=periods, tz="UTC")
    else:
        index = pd.date_range("2025-01-02 14:30:00", periods=periods, freq=freq, tz="UTC")
    mu = 0.00025 if freq == "B" else 0.00005
    sigma = 0.018 if freq == "B" else 0.004
    shocks = rng.normal(mu, sigma, size=periods)
    close = START_PRICES[symbol] * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, size=periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, size=periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, size=periods))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(1_500_000, 18_000_000, size=periods).astype(float)
    ingestion = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
            "source": "sample",
            "ingestion_time": ingestion,
        }
    )
    return frame[OHLCV_COLUMNS]


def main() -> None:
    set_global_seed(settings.random_seed)
    sample_dir = settings.data_root_path / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, symbol in enumerate(SUPPORTED_SYMBOLS):
        daily = simulate_ohlcv(symbol, periods=1260, freq="B", seed_offset=index * 10)
        hourly = simulate_ohlcv(symbol, periods=1100, freq="h", seed_offset=index * 10 + 1)
        m15 = simulate_ohlcv(symbol, periods=1100, freq="15min", seed_offset=index * 10 + 2)
        m5 = simulate_ohlcv(symbol, periods=1100, freq="5min", seed_offset=index * 10 + 3)
        for interval, frame in [("1d", daily), ("1h", hourly), ("15m", m15), ("5m", m5)]:
            csv_path = sample_dir / f"{symbol}_{interval}.csv"
            parquet_path = sample_dir / f"{symbol}_{interval}.parquet"
            frame.to_csv(csv_path, index=False)
            frame.to_parquet(parquet_path, index=False)
            logger.info("Wrote %s rows to %s", len(frame), csv_path.name)
    logger.info("Sample data generated in %s", sample_dir)


if __name__ == "__main__":
    main()
