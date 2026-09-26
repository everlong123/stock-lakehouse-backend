"""Gold layer: analytics-ready features and targets without look-ahead leakage."""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.features.feature_engineering import build_gold_features
from app.lakehouse.parquet_manager import split_by_partition
from app.lakehouse.spark_session import get_spark_session, spark_enabled
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)


class GoldLayer:
    """Feature and target zone consumed by TA, ML, DL, backtesting, and dashboards."""

    layer_name = "gold"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def transform(self, silver_frame: pd.DataFrame) -> pd.DataFrame:
        """Build Gold features. Spark is used for column-level transforms when enabled."""
        if silver_frame.empty:
            raise DataValidationError("Silver frame is empty; cannot build Gold.")
        spark = get_spark_session() if spark_enabled() else None
        if spark is not None:
            return self._transform_spark(silver_frame, spark)
        return build_gold_features(silver_frame)

    def write(self, gold_frame: pd.DataFrame) -> dict:
        partitions = split_by_partition(gold_frame)
        paths = []
        for relative_path, group in partitions.items():
            paths.append(self.storage.write_parquet(self.layer_name, relative_path, group))
        logger.info("Gold write complete: %s rows, %s partitions", len(gold_frame), len(paths))
        return {"paths": paths, "records": int(len(gold_frame))}

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

    def _transform_spark(self, frame: pd.DataFrame, spark) -> pd.DataFrame:
        """Apply Spark numeric casting then reuse Pandas feature builder for window features."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, StringType, TimestampType

        sdf = spark.createDataFrame(frame)
        sdf = sdf.withColumn("symbol", F.upper(F.col("symbol").cast(StringType())))
        sdf = sdf.withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            sdf = sdf.withColumn(col, F.col(col).cast(DoubleType()))
        pandas_frame = sdf.orderBy("symbol", "timestamp").toPandas()
        pandas_frame["timestamp"] = pd.to_datetime(pandas_frame["timestamp"], utc=True)
        return build_gold_features(pandas_frame)
