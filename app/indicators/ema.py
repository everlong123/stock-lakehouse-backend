"""Exponential moving average."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, window: int) -> pd.Series:
    """Return the exponential moving average of `series` using span=`window`."""
    if window <= 0:
        raise ValueError("EMA window must be a positive integer.")
    return series.astype(float).ewm(span=window, adjust=False, min_periods=window).mean()
