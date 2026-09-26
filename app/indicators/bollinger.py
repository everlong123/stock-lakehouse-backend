"""Bollinger Bands."""

from __future__ import annotations

import pandas as pd

from app.indicators.sma import sma


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Return middle, upper, and lower Bollinger Bands."""
    close = series.astype(float)
    middle = sma(close, window)
    rolling_std = close.rolling(window=window, min_periods=window).std()
    upper = middle + std_multiplier * rolling_std
    lower = middle - std_multiplier * rolling_std
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
        },
        index=series.index,
    )
