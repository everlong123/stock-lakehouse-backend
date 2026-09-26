"""Spark session factory with a safe fallback when Spark is disabled."""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_SPARK = None


def spark_enabled() -> bool:
    """Return True when USE_SPARK=true."""
    return bool(settings.use_spark)


def get_spark_session() -> Any | None:
    """Create a local SparkSession when enabled. Returns None otherwise."""
    global _SPARK
    if not spark_enabled():
        return None
    if _SPARK is not None:
        return _SPARK
    try:
        from pyspark.sql import SparkSession

        _SPARK = (
            SparkSession.builder.appName(settings.spark_app_name)
            .master(settings.spark_master)
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        logger.info("SparkSession started: %s", settings.spark_master)
        return _SPARK
    except Exception as exc:
        logger.warning("Spark is enabled but could not start (%s). Using Pandas.", exc)
        return None
