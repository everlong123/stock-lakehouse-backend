"""Data quality gate between Silver and Gold."""

from __future__ import annotations

from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.lakehouse.silver import SilverLayer
from app.pipelines.validation import validate_ohlc_frame

logger = get_logger(__name__)


def build_quality_report(symbol: str) -> dict:
    """Produce a quality report from Silver data."""
    frame = SilverLayer().read(symbol)
    if frame.empty:
        report = {
            "symbol": symbol.upper(),
            "record_count": 0,
            "duplicate_count": 0,
            "missing_count": 0,
            "invalid_ohlc_count": 0,
            "invalid_volume_count": 0,
            "min_timestamp": None,
            "max_timestamp": None,
            "quality_status": "failed",
        }
        logger.error("Quality check failed for %s: Silver is empty.", symbol)
        return report
    counts = validate_ohlc_frame(frame)
    critical = counts["invalid_ohlc_count"] > 0 or counts["row_count"] == 0
    report = {
        "symbol": symbol.upper(),
        "record_count": counts["row_count"],
        "duplicate_count": counts["duplicate_count"],
        "missing_count": counts["missing_count"],
        "invalid_ohlc_count": counts["invalid_ohlc_count"],
        "invalid_volume_count": counts["invalid_volume_count"],
        "min_timestamp": frame["timestamp"].min().isoformat(),
        "max_timestamp": frame["timestamp"].max().isoformat(),
        "quality_status": "failed" if critical else "passed",
    }
    logger.info("Quality report for %s: %s", symbol, report)
    return report


def assert_quality_passed(report: dict) -> None:
    """Block Gold builds when critical validation fails."""
    if report.get("quality_status") == "failed":
        raise DataValidationError(
            f"Critical data quality failure for {report.get('symbol')}: {report}"
        )
