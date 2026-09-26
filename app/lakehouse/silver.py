"""Silver layer: cleaned, validated, timezone-normalized OHLCV."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.lakehouse.parquet_manager import split_by_partition
from app.lakehouse.spark_session import get_spark_session, spark_enabled
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)


class SilverLayer:
    """Cleaned zone. Invalid records are reported, never silently dropped without counts."""

    layer_name = "silver"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def transform(self, bronze_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Clean Bronze data using Spark when enabled, otherwise Pandas."""
        if bronze_frame.empty:
            raise DataValidationError("Bronze frame is empty; cannot build Silver.")
        spark = get_spark_session() if spark_enabled() else None
        if spark is not None:
            return self._transform_spark(bronze_frame, spark)
        return self._transform_pandas(bronze_frame)

    def write(self, silver_frame: pd.DataFrame, quality_report: dict) -> dict:
        """Overwrite Silver partitions for the symbols present in the frame."""
        partitions = split_by_partition(silver_frame)
        paths = []
        for relative_path, group in partitions.items():
            paths.append(self.storage.write_parquet(self.layer_name, relative_path, group))
        for symbol in silver_frame["symbol"].unique():
            self.storage.write_json(
                self.layer_name,
                f"symbol={symbol}/_quality.json",
                quality_report,
            )
        return {"paths": paths, "records": int(len(silver_frame))}

    def read(self, symbol: str) -> pd.DataFrame:
        frame = self.storage.read_prefix(self.layer_name, f"symbol={symbol.upper()}")
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        return frame.sort_values("timestamp").reset_index(drop=True)

    def record_count(self, symbol: str | None = None) -> int:
        prefix = f"symbol={symbol.upper()}" if symbol else ""
        return int(len(self.storage.read_prefix(self.layer_name, prefix)))

    def _transform_pandas(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        working = frame.copy()
        original_count = len(working)
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working["ingestion_time"] = pd.to_datetime(working["ingestion_time"], utc=True)
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            working[col] = pd.to_numeric(working[col], errors="coerce")
        working["symbol"] = working["symbol"].astype(str).str.upper()

        missing_mask = working[["open", "high", "low", "close", "volume", "timestamp"]].isna().any(axis=1)
        duplicate_mask = working.duplicated(subset=["symbol", "timestamp"], keep="last")
        invalid_ohlc = ~(
            (working["high"] >= working["open"])
            & (working["high"] >= working["close"])
            & (working["low"] <= working["open"])
            & (working["low"] <= working["close"])
            & (working["high"] >= working["low"])
        )
        invalid_volume = working["volume"] < 0
        invalid_mask = missing_mask | invalid_ohlc | invalid_volume

        error_frame = working[invalid_mask | duplicate_mask].copy()
        error_frame["error_reason"] = ""
        error_frame.loc[missing_mask, "error_reason"] = error_frame.loc[missing_mask, "error_reason"] + "missing;"
        error_frame.loc[invalid_ohlc, "error_reason"] = error_frame.loc[invalid_ohlc, "error_reason"] + "invalid_ohlc;"
        error_frame.loc[invalid_volume, "error_reason"] = (
            error_frame.loc[invalid_volume, "error_reason"] + "invalid_volume;"
        )
        error_frame.loc[duplicate_mask, "error_reason"] = (
            error_frame.loc[duplicate_mask, "error_reason"] + "duplicate;"
        )

        valid = working[~invalid_mask].drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        valid = valid.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        if not error_frame.empty:
            self.storage.write_parquet(
                self.layer_name,
                f"_errors/errors_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.parquet",
                error_frame,
            )
            logger.warning("Silver rejected %s invalid/duplicate rows", len(error_frame))

        quality = {
            "engine": "pandas",
            "record_count": int(len(valid)),
            "source_count": int(original_count),
            "duplicate_count": int(duplicate_mask.sum()),
            "missing_count": int(missing_mask.sum()),
            "invalid_ohlc_count": int(invalid_ohlc.sum()),
            "invalid_volume_count": int(invalid_volume.sum()),
            "error_count": int(len(error_frame)),
            "min_timestamp": valid["timestamp"].min().isoformat() if not valid.empty else None,
            "max_timestamp": valid["timestamp"].max().isoformat() if not valid.empty else None,
            "quality_status": "passed" if not valid.empty and invalid_ohlc.sum() == 0 else "warning",
        }
        if valid.empty:
            quality["quality_status"] = "failed"
        return valid, quality

    def _transform_spark(self, frame: pd.DataFrame, spark) -> tuple[pd.DataFrame, dict]:
        """Spark implementation with the same output schema as Pandas."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, StringType, TimestampType

        sdf = spark.createDataFrame(frame)
        sdf = (
            sdf.withColumn("symbol", F.upper(F.col("symbol").cast(StringType())))
            .withColumn("timestamp", F.to_utc_timestamp(F.col("timestamp").cast(TimestampType()), "UTC"))
            .withColumn("ingestion_time", F.col("ingestion_time").cast(TimestampType()))
        )
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            sdf = sdf.withColumn(col, F.col(col).cast(DoubleType()))

        invalid = (
            F.col("open").isNull()
            | F.col("high").isNull()
            | F.col("low").isNull()
            | F.col("close").isNull()
            | F.col("volume").isNull()
            | (F.col("high") < F.col("open"))
            | (F.col("high") < F.col("close"))
            | (F.col("low") > F.col("open"))
            | (F.col("low") > F.col("close"))
            | (F.col("high") < F.col("low"))
            | (F.col("volume") < 0)
        )
        valid_sdf = sdf.filter(~invalid).dropDuplicates(["symbol", "timestamp"]).orderBy("symbol", "timestamp")
        error_count = sdf.filter(invalid).count()
        pandas_frame = valid_sdf.toPandas()
        if "timestamp" in pandas_frame.columns:
            pandas_frame["timestamp"] = pd.to_datetime(pandas_frame["timestamp"], utc=True)
        quality = {
            "engine": "spark",
            "record_count": int(len(pandas_frame)),
            "source_count": int(len(frame)),
            "duplicate_count": int(len(frame) - sdf.dropDuplicates(["symbol", "timestamp"]).count()),
            "missing_count": 0,
            "invalid_ohlc_count": int(error_count),
            "invalid_volume_count": 0,
            "error_count": int(error_count),
            "min_timestamp": pandas_frame["timestamp"].min().isoformat() if not pandas_frame.empty else None,
            "max_timestamp": pandas_frame["timestamp"].max().isoformat() if not pandas_frame.empty else None,
            "quality_status": "passed" if not pandas_frame.empty else "failed",
        }
        return pandas_frame, quality
