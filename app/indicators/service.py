"""Indicator computation service used by APIs and the Gold layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.indicators.bollinger import bollinger_bands
from app.indicators.ema import ema
from app.indicators.macd import macd
from app.indicators.rsi import rsi
from app.indicators.sma import sma


def _scalar(value) -> float | None:
    if isinstance(value, pd.Series):
        value = value.iloc[-1] if len(value) else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


@dataclass
class IndicatorConfig:
    """User-configurable indicator parameters."""

    sma_windows: tuple[int, ...] = (5, 10, 20, 50)
    ema_windows: tuple[int, ...] = (12, 26)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0


def add_indicators(frame: pd.DataFrame, config: IndicatorConfig | None = None) -> pd.DataFrame:
    """Attach SMA, EMA, RSI, MACD, and Bollinger columns to an OHLCV frame."""
    if "close" not in frame.columns:
        raise ValueError("Indicator calculation requires a close column.")
    config = config or IndicatorConfig()
    result = frame.copy()
    close = result["close"]
    for window in config.sma_windows:
        result[f"sma_{window}"] = sma(close, window)
    for window in config.ema_windows:
        result[f"ema_{window}"] = ema(close, window)
    result["rsi_14"] = rsi(close, config.rsi_period)
    macd_frame = macd(close, config.macd_fast, config.macd_slow, config.macd_signal)
    result = pd.concat([result, macd_frame], axis=1)
    bands = bollinger_bands(close, config.bb_window, config.bb_std)
    result = pd.concat([result, bands], axis=1)
    return result


def describe_indicators(latest: pd.Series) -> dict[str, str]:
    """Return a descriptive (non-advisory) summary of the latest indicator snapshot."""
    trend = "insufficient data"
    momentum = "insufficient data"
    volatility = "insufficient data"
    close = _scalar(latest.get("close"))
    sma20 = _scalar(latest.get("sma_20"))
    sma50 = _scalar(latest.get("sma_50"))
    rsi_value = _scalar(latest.get("rsi_14"))
    bb_upper = _scalar(latest.get("bb_upper"))
    bb_lower = _scalar(latest.get("bb_lower"))

    if close is not None and sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            trend = "Price is above SMA20 and SMA20 is above SMA50 (uptrend structure)."
        elif close < sma20 < sma50:
            trend = "Price is below SMA20 and SMA20 is below SMA50 (downtrend structure)."
        else:
            trend = "Moving averages are mixed; no dominant trend structure."

    if rsi_value is not None:
        if rsi_value >= 70:
            momentum = f"RSI is {rsi_value:.2f}, in the historically elevated zone."
        elif rsi_value <= 30:
            momentum = f"RSI is {rsi_value:.2f}, in the historically depressed zone."
        else:
            momentum = f"RSI is {rsi_value:.2f}, in a mid-range zone."

    if close is not None and bb_upper is not None and bb_lower is not None:
        width = bb_upper - bb_lower
        if close > bb_upper:
            volatility = "Close is above the upper Bollinger Band (expanded deviation)."
        elif close < bb_lower:
            volatility = "Close is below the lower Bollinger Band (expanded deviation)."
        else:
            volatility = f"Close is inside the Bollinger Band. Band width is {width:.2f}."

    return {"trend": trend, "momentum": momentum, "volatility": volatility}
