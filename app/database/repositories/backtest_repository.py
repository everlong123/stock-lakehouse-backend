"""Persistence helpers for backtests."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models.backtest_run import BacktestRun


class BacktestRepository:
    """CRUD operations for backtest_runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: dict[str, Any]) -> BacktestRun:
        record = BacktestRun(**payload)
        self.session.add(record)
        self.session.flush()
        return record

    def list_history(self, symbol: str | None = None, limit: int = 30) -> list[BacktestRun]:
        stmt = select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
        if symbol:
            stmt = (
                select(BacktestRun)
                .where(BacktestRun.symbol == symbol.upper())
                .order_by(BacktestRun.created_at.desc())
                .limit(limit)
            )
        return list(self.session.scalars(stmt).all())

    def latest(self, symbol: str | None = None) -> BacktestRun | None:
        stmt = select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(1)
        if symbol:
            stmt = (
                select(BacktestRun)
                .where(BacktestRun.symbol == symbol.upper())
                .order_by(BacktestRun.created_at.desc())
                .limit(1)
            )
        return self.session.scalars(stmt).first()
