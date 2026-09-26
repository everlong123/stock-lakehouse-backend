"""Forecasting request schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str = Field(min_length=1, max_length=16)
    model_name: str = Field(description="linear_regression | arima | lstm")
    horizon: int = Field(default=1, ge=1, le=30)
    sequence_length: int | None = 60
    hidden_size: int | None = 64
    num_layers: int | None = 2
    epochs: int | None = 20
    arima_p: int | None = 5
    arima_d: int | None = 1
    arima_q: int | None = 0


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    symbol: str = Field(min_length=1, max_length=16)
    model_name: str
    horizon: int = Field(default=5, ge=1, le=60)
