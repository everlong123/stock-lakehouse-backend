"""MACD indicator: fast EMA, slow EMA, and signal line."""

from __future__ import annotations

import pandas as pd

from app.indicators.ema import ema


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Return a DataFrame with macd, macd_signal, and macd_hist columns."""
    close = series.astype(float)
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        },
        index=series.index,
    )
