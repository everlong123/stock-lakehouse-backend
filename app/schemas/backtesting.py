"""Backtesting request schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=16)
    strategy: str = Field(description="ma_crossover | rsi_strategy")
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: float = Field(default=10000, gt=0)
    transaction_fee: float = Field(default=0.001, ge=0, le=0.05)
    slippage: float = Field(default=0.0005, ge=0, le=0.05)
    short_window: int = 20
    long_window: int = 50
    rsi_period: int = 14
    lower_threshold: float = 30
    upper_threshold: float = 70
