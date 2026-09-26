"""End-to-end Bronze → Silver → Gold runner."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.logging_config import get_logger
from app.pipelines.feature_pipeline import build_gold_layer
from app.pipelines.quality import assert_quality_passed, build_quality_report
from app.pipelines.transformation import transform_to_silver

logger = get_logger(__name__)


def run_symbol_pipeline(
    symbol: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    source_name: str | None = None,
) -> dict:
    """Run the full medallion pipeline for one symbol. Tasks are idempotent on rerun."""
    started = datetime.now(timezone.utc)
    silver_meta = transform_to_silver(symbol)
    quality = build_quality_report(symbol)
    assert_quality_passed(quality)
    gold_meta = build_gold_layer(symbol)
    finished = datetime.now(timezone.utc)
    result = {
        "pipeline_name": "stock_lakehouse_pipeline",
        "symbol": symbol.upper(),
        "status": "success",
        "start_time": started,
        "end_time": finished,
        "records_processed": gold_meta.get("records", 0),
        "error_count": silver_meta.get("quality", {}).get("error_count", 0),
        "message": "Pipeline completed.",
        "silver": silver_meta,
        "quality": quality,
        "gold": gold_meta,
    }
    logger.info("Pipeline finished for %s in %s seconds", symbol, (finished - started).total_seconds())
    return result
