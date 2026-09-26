"""Forecast model training metadata."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ModelRun(Base):
    """One trained forecasting model and its evaluation metrics."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    train_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    train_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    validation_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    validation_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    test_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    test_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    features: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[str | None] = mapped_column(Text, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    directional_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
