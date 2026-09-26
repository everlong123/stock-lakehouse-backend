"""Technical analysis schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndicatorRequest(BaseModel):
    interval: str = "1d"
    sma: bool = True
    ema: bool = True
    rsi: bool = True
    macd: bool = True
    bollinger: bool = True
    sma_windows: list[int] = Field(default_factory=lambda: [20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
