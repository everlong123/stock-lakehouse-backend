"""Pipeline service used by the API."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.database.repositories.pipeline_repository import PipelineRepository
from app.database.session import check_database_connection
from app.pipelines.pipeline_runner import run_symbol_pipeline

logger = get_logger(__name__)


class PipelineService:
    def run(self, symbol: str, interval: str = "1d", source: str | None = None, db: Session | None = None) -> dict:
        result = run_symbol_pipeline(symbol=symbol, interval=interval, source_name=source)
        if db is not None and check_database_connection():
            try:
                PipelineRepository(db).create(
                    {
                        "pipeline_name": result["pipeline_name"],
                        "symbol": result["symbol"],
                        "status": result["status"],
                        "start_time": result["start_time"].replace(tzinfo=None)
                        if isinstance(result["start_time"], datetime)
                        else None,
                        "end_time": result["end_time"].replace(tzinfo=None)
                        if isinstance(result["end_time"], datetime)
                        else None,
                        "records_processed": result["records_processed"],
                        "error_count": result["error_count"],
                        "message": result["message"],
                    }
                )
            except Exception as exc:
                logger.warning("Unable to persist pipeline run: %s", exc)
        return result
