"""Supervised targets. All targets are shifted forward so they never leak into features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add next-bar close, return, and direction. The last row is NaN by design."""
    result = frame.copy()
    close = result["close"].astype(float)
    result["target_close_next"] = close.shift(-1)
    result["target_return_next"] = close.pct_change().shift(-1)
    result["target_direction_next"] = np.where(
        result["target_return_next"] > 0,
        1,
        np.where(result["target_return_next"].isna(), np.nan, 0),
    )
    return result
