"""Price-derived features computed only from current and past values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add returns, log returns, and price/volume changes."""
    result = frame.copy()
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    result["return"] = close.pct_change()
    result["log_return"] = np.log(close / close.shift(1))
    result["price_change"] = close.diff()
    result["volume_change"] = volume.pct_change()
    result["rolling_std_20"] = result["return"].rolling(window=20, min_periods=20).std()
    result["rolling_min_20"] = close.rolling(window=20, min_periods=20).min()
    result["rolling_max_20"] = close.rolling(window=20, min_periods=20).max()
    result["volume_ma_20"] = volume.rolling(window=20, min_periods=20).mean()
    result["close_lag_1"] = close.shift(1)
    result["close_lag_2"] = close.shift(2)
    result["return_lag_1"] = result["return"].shift(1)
    result["volume_lag_1"] = volume.shift(1)
    return result
