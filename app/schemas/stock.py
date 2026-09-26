"""Stock and pipeline request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OHLCVPoint(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float
    source: str | None = None


class StockQuery(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    interval: str = "1d"


class PipelineRunRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=16)
    interval: str = "1d"
    source: str | None = None
