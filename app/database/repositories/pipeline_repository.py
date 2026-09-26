"""Persistence helpers for pipeline runs."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models.pipeline_run import PipelineRun


class PipelineRepository:
    """CRUD operations for pipeline_runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: dict[str, Any]) -> PipelineRun:
        record = PipelineRun(**payload)
        self.session.add(record)
        self.session.flush()
        return record

    def list_recent(self, symbol: str | None = None, limit: int = 20) -> list[PipelineRun]:
        stmt = select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(limit)
        if symbol:
            stmt = (
                select(PipelineRun)
                .where(PipelineRun.symbol == symbol.upper())
                .order_by(PipelineRun.created_at.desc())
                .limit(limit)
            )
        return list(self.session.scalars(stmt).all())

    def latest(self, symbol: str | None = None) -> PipelineRun | None:
        stmt = select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(1)
        if symbol:
            stmt = (
                select(PipelineRun)
                .where(PipelineRun.symbol == symbol.upper())
                .order_by(PipelineRun.created_at.desc())
                .limit(1)
            )
        return self.session.scalars(stmt).first()
