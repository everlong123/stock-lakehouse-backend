"""OHLCV validation helpers used by pipelines and tests."""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import DataValidationError


def validate_ohlc_frame(frame: pd.DataFrame) -> dict[str, int]:
    """Count validation issues without mutating the input."""
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required if col not in frame.columns]
    if missing_cols:
        raise DataValidationError(f"Validation missing columns: {missing_cols}")

    missing_count = int(frame[required].isna().any(axis=1).sum())
    duplicate_count = int(frame.duplicated(subset=["symbol", "timestamp"]).sum())
    invalid_ohlc_count = int(
        (
            ~(
                (frame["high"] >= frame["open"])
                & (frame["high"] >= frame["close"])
                & (frame["low"] <= frame["open"])
                & (frame["low"] <= frame["close"])
                & (frame["high"] >= frame["low"])
            )
        ).sum()
    )
    invalid_volume_count = int((frame["volume"] < 0).sum())
    return {
        "row_count": int(len(frame)),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "invalid_ohlc_count": invalid_ohlc_count,
        "invalid_volume_count": invalid_volume_count,
    }
