"""Simple moving average."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Return the simple moving average of `series` over `window` periods."""
    if window <= 0:
        raise ValueError("SMA window must be a positive integer.")
    return series.astype(float).rolling(window=window, min_periods=window).mean()
