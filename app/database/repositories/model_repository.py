"""Persistence helpers for trained forecast models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models.model_run import ModelRun


class ModelRepository:
    """CRUD operations for model_runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: dict[str, Any]) -> ModelRun:
        record = ModelRun(**payload)
        self.session.add(record)
        self.session.flush()
        return record

    def list_by_symbol(self, symbol: str, limit: int = 20) -> list[ModelRun]:
        stmt = (
            select(ModelRun)
            .where(ModelRun.symbol == symbol.upper())
            .order_by(ModelRun.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def latest_by_symbol_and_model(self, symbol: str, model_name: str) -> ModelRun | None:
        stmt = (
            select(ModelRun)
            .where(ModelRun.symbol == symbol.upper(), ModelRun.model_name == model_name)
            .order_by(ModelRun.created_at.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).first()

    def latest(self, symbol: str | None = None) -> ModelRun | None:
        stmt = select(ModelRun).order_by(ModelRun.created_at.desc()).limit(1)
        if symbol:
            stmt = (
                select(ModelRun)
                .where(ModelRun.symbol == symbol.upper())
                .order_by(ModelRun.created_at.desc())
                .limit(1)
            )
        return self.session.scalars(stmt).first()

    def comparison_rows(self, symbol: str) -> list[ModelRun]:
        """Return the latest run per model name for a symbol."""
        rows = self.list_by_symbol(symbol, limit=50)
        latest: dict[str, ModelRun] = {}
        for row in rows:
            if row.model_name not in latest:
                latest[row.model_name] = row
        return list(latest.values())
