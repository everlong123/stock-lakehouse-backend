"""Bronze layer: raw ingested OHLCV close to the source schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.core.constants import OHLCV_COLUMNS
from app.core.exceptions import DataValidationError, StorageError
from app.core.logging_config import get_logger
from app.lakehouse.parquet_manager import split_by_partition
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)

BRONZE_SCHEMA = {
    "symbol": "string",
    "timestamp": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "float64",
    "source": "string",
    "ingestion_time": "datetime64[ns, UTC]",
}


class BronzeLayer:
    """Append-only raw lakehouse zone with partition, lineage, and duplicate checks."""

    layer_name = "bronze"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def validate_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and types are coercible."""
        missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
        if missing:
            raise DataValidationError(f"Bronze schema missing columns: {missing}")
        result = frame[OHLCV_COLUMNS].copy()
        result["symbol"] = result["symbol"].astype(str).str.upper()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        result["ingestion_time"] = pd.to_datetime(result["ingestion_time"], utc=True)
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        result["source"] = result["source"].astype(str)
        return result

    def append(self, frame: pd.DataFrame, lineage: dict | None = None) -> dict:
        """Append rows to partitioned Parquet files. Duplicates are counted, not silently ignored."""
        if frame.empty:
            raise DataValidationError("Cannot write empty frame to Bronze.")
        clean = self.validate_schema(frame)
        clean["ingestion_time"] = clean["ingestion_time"].fillna(pd.Timestamp.now(tz="UTC"))
        duplicate_count = int(clean.duplicated(subset=["symbol", "timestamp"]).sum())
        partitions = split_by_partition(clean)
        written_paths: list[str] = []
        records_written = 0
        for relative_path, group in partitions.items():
            existing = pd.DataFrame()
            if self.storage.exists(self.layer_name, relative_path):
                existing = self.storage.read_parquet(self.layer_name, relative_path)
            combined = pd.concat([existing, group], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            dropped = before - len(combined)
            duplicate_count += int(dropped)
            path = self.storage.write_parquet(self.layer_name, relative_path, combined)
            written_paths.append(path)
            records_written += len(group)
        metadata = {
            "layer": self.layer_name,
            "records_received": int(len(clean)),
            "records_written": records_written,
            "duplicate_count": duplicate_count,
            "partitions": list(partitions.keys()),
            "paths": written_paths,
            "lineage": lineage or {},
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
            "storage_backend": self.storage.backend_name,
        }
        symbols = sorted(clean["symbol"].unique().tolist())
        for symbol in symbols:
            self.storage.write_json(
                self.layer_name,
                f"symbol={symbol}/_lineage.json",
                metadata,
            )
        logger.info(
            "Bronze append complete: received=%s duplicates=%s paths=%s",
            len(clean),
            duplicate_count,
            len(written_paths),
        )
        return metadata

    def read(self, symbol: str) -> pd.DataFrame:
        """Read all Bronze partitions for a symbol."""
        frame = self.storage.read_prefix(self.layer_name, f"symbol={symbol.upper()}")
        if frame.empty:
            return frame
        return self.validate_schema(frame).sort_values("timestamp").reset_index(drop=True)

    def record_count(self, symbol: str | None = None) -> int:
        prefix = f"symbol={symbol.upper()}" if symbol else ""
        frame = self.storage.read_prefix(self.layer_name, prefix)
        return int(len(frame))
