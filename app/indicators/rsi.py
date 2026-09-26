"""Relative Strength Index using Wilder's smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return RSI values in [0, 100] for the given close series."""
    if period <= 0:
        raise ValueError("RSI period must be a positive integer.")
    close = series.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    result = result.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    return result.astype(float)
