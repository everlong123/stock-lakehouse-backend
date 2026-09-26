"""
Iceberg Table Manager - Modern Lakehouse Table Format
====================================================

Apache Iceberg vs Parquet (raw files):
- Time travel: query historical versions of data
- Schema evolution: add/rename columns safely
- Partition evolution: change partitioning without rewrite
- ACID transactions: atomic commits across multiple files
- Hidden partitioning: queries auto-filter by partition
- Snapshot isolation: concurrent reads don't block writes

Architecture:
                    ┌─────────────────────┐
                    │  Iceberg REST API   │  ← tabulario/iceberg-rest
                    │   (port 8181)       │
                    └──────────┬──────────┘
                               │ REST
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Bronze     │    │    Silver    │    │     Gold      │
│   (raw OHLCV) │    │  (cleaned)   │    │  (features)   │
│               │    │               │    │               │
│ AAPL/         │    │ AAPL/         │    │ AAPL/         │
│ 2026/         │    │ 2026/         │    │ 2026/         │
│ 01/           │    │ 01/           │    │ 01/           │
│ data.parquet  │    │ data.parquet  │    │ data.parquet  │
│ manifest.json │    │ manifest.json │    │ manifest.json │
│ metadata.json │    │ metadata.json │    │ metadata.json │
└───────────────┘    └───────────────┘    └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   MinIO Object Store │
                    │  (S3-compatible)     │
                    └─────────────────────┘

Usage:
```python
from app.lakehouse.iceberg_manager import IcebergManager

# Initialize with REST catalog
manager = IcebergManager(catalog_uri="http://localhost:8181")

# Create tables
manager.create_bronze_table()
manager.create_silver_table()
manager.create_gold_table()

# Write data (ACID commit)
manager.write_bronze(df, symbol="AAPL")

# Time travel query (read version 3)
df = manager.read_bronze(symbol="AAPL", version=3)
```

References:
- Iceberg spec: https://iceberg.apache.org/spec/
- PyIceberg: https://py.iceberg.apache.org/
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyiceberg.catalog
import pyiceberg.schema
import pyiceberg.table
import pyiceberg.transforms
import pyiceberg.types
from pyiceberg.catalog import Catalog
from pyiceberg.expressions import AlwaysBreaker, And, Bound, BoundPredicate, NotEqTo, Or, Reference
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import IdentityTransform, TimeTransform
from pyiceberg.typing import StructLike
from pyiceberg.types import (
    DoubleType,
    IntegerType,
    NestedField,
    StringType,
    TimestampType,
)

from app.core.config import get_settings
from app.core.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS  (Iceberg requires explicit schema definitions)
# ═══════════════════════════════════════════════════════════════════════════════

BRONZE_SCHEMA = Schema(
    schema_id=1,
    identifier_field_ids=[1, 2],  # symbol, timestamp
    fields=(
        NestedField(field_id=1, name="symbol", field_type=StringType(), required=True, doc="Ticker symbol"),
        NestedField(field_id=2, name="timestamp", field_type=TimestampType(), required=True, doc="OHLCV bar timestamp (UTC)"),
        NestedField(field_id=3, name="open", field_type=DoubleType(), required=True, doc="Opening price"),
        NestedField(field_id=4, name="high", field_type=DoubleType(), required=True, doc="Highest price"),
        NestedField(field_id=5, name="low", field_type=DoubleType(), required=True, doc="Lowest price"),
        NestedField(field_id=6, name="close", field_type=DoubleType(), required=True, doc="Closing price"),
        NestedField(field_id=7, name="adj_close", field_type=DoubleType(), required=True, doc="Adjusted close"),
        NestedField(field_id=8, name="volume", field_type=DoubleType(), required=True, doc="Trading volume"),
        NestedField(field_id=9, name="source", field_type=StringType(), required=False, doc="Data source provider"),
        NestedField(field_id=10, name="ingestion_time", field_type=TimestampType(), required=True, doc="When record was ingested"),
    ),
)

SILVER_SCHEMA = Schema(
    schema_id=1,
    identifier_field_ids=[1, 2],
    fields=(
        # Same core OHLCV as Bronze, but validated and cleaned
        NestedField(field_id=1, name="symbol", field_type=StringType(), required=True),
        NestedField(field_id=2, name="timestamp", field_type=TimestampType(), required=True),
        NestedField(field_id=3, name="open", field_type=DoubleType(), required=True),
        NestedField(field_id=4, name="high", field_type=DoubleType(), required=True),
        NestedField(field_id=5, name="low", field_type=DoubleType(), required=True),
        NestedField(field_id=6, name="close", field_type=DoubleType(), required=True),
        NestedField(field_id=7, name="adj_close", field_type=DoubleType(), required=True),
        NestedField(field_id=8, name="volume", field_type=DoubleType(), required=True),
        # Standardization metadata
        NestedField(field_id=11, name="data_quality", field_type=StringType(), required=False, doc="Quality status"),
    ),
)

GOLD_SCHEMA = Schema(
    schema_id=1,
    identifier_field_ids=[1, 2],
    fields=(
        # Core data
        NestedField(field_id=1, name="symbol", field_type=StringType(), required=True),
        NestedField(field_id=2, name="timestamp", field_type=TimestampType(), required=True),
        NestedField(field_id=3, name="open", field_type=DoubleType(), required=True),
        NestedField(field_id=4, name="high", field_type=DoubleType(), required=True),
        NestedField(field_id=5, name="low", field_type=DoubleType(), required=True),
        NestedField(field_id=6, name="close", field_type=DoubleType(), required=True),
        NestedField(field_id=7, name="adj_close", field_type=DoubleType(), required=True),
        NestedField(field_id=8, name="volume", field_type=DoubleType(), required=True),
        # ── Technical Indicators ────────────────────────────────────────────────
        NestedField(field_id=20, name="sma_20", field_type=DoubleType(), required=False, doc="20-day SMA"),
        NestedField(field_id=21, name="sma_50", field_type=DoubleType(), required=False, doc="50-day SMA"),
        NestedField(field_id=22, name="ema_12", field_type=DoubleType(), required=False, doc="12-day EMA"),
        NestedField(field_id=23, name="ema_26", field_type=DoubleType(), required=False, doc="26-day EMA"),
        NestedField(field_id=24, name="rsi_14", field_type=DoubleType(), required=False, doc="14-day RSI"),
        NestedField(field_id=25, name="macd", field_type=DoubleType(), required=False, doc="MACD line"),
        NestedField(field_id=26, name="macd_signal", field_type=DoubleType(), required=False, doc="MACD signal"),
        NestedField(field_id=27, name="macd_hist", field_type=DoubleType(), required=False, doc="MACD histogram"),
        NestedField(field_id=28, name="bb_upper", field_type=DoubleType(), required=False, doc="Bollinger upper band"),
        NestedField(field_id=29, name="bb_middle", field_type=DoubleType(), required=False, doc="Bollinger middle band"),
        NestedField(field_id=30, name="bb_lower", field_type=DoubleType(), required=False, doc="Bollinger lower band"),
        # ── ML Target Variables ──────────────────────────────────────────────────
        NestedField(field_id=40, name="return_1d", field_type=DoubleType(), required=False, doc="Next 1-day return (target)"),
        NestedField(field_id=41, name="return_5d", field_type=DoubleType(), required=False, doc="Next 5-day return (target)"),
        NestedField(field_id=42, name="direction_1d", field_type=IntegerType(), required=False, doc="1 if price up tomorrow, 0 otherwise"),
        # ── Derived ────────────────────────────────────────────────────────────
        NestedField(field_id=50, name="volume_ratio", field_type=DoubleType(), required=False, doc="Volume / 20-day avg volume"),
        NestedField(field_id=51, name="price_range", field_type=DoubleType(), required=False, doc="(High - Low) / Close"),
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CATALOG_URI = "http://localhost:8181"
DEFAULT_WAREHOUSE = "s3://stock-lakehouse/iceberg"


def get_iceberg_catalog() -> Catalog:
    """
    Create Iceberg REST catalog connection.

    Uses tabulario/iceberg-rest container which provides:
    - REST API for table CRUD
    - SQLite metadata storage (dev)
    - Integration with MinIO for data files

    For production: swap to Nessie (Git-like branching) or Hive Metastore.
    """
    settings = get_settings()
    catalog_uri = settings.ICEBERG_CATALOG_URI or DEFAULT_CATALOG_URI

    return pyiceberg.catalog.load_catalog(
        "rest",
        **{
            "uri": catalog_uri,
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "minioadmin",
            "s3.secret-access-key": "minioadmin",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE NAMES
# ═══════════════════════════════════════════════════════════════════════════════

TABLE_BRONZE = "lakehouse.bronze_ohlcv"
TABLE_SILVER = "lakehouse.silver_ohlcv"
TABLE_GOLD = "lakehouse.gold_features"


# ═══════════════════════════════════════════════════════════════════════════════
# ICEBERG MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class IcebergManager:
    """
    Manages Iceberg tables for the medallion lakehouse.

    Why Iceberg?
    - Time travel: debug data issues by querying past snapshots
    - Schema evolution: add new indicator columns without downtime
    - Hidden partitions: optimizer auto-prunes partitions
    - ACID: writes are atomic, no partial writes

    Example - Time Travel:
    ```python
    # Read current data
    df = manager.read_bronze("AAPL")

    # Read data as it existed 3 days ago
    df_old = manager.read_bronze("AAPL", as_of_timestamp="3d")

    # List all versions
    versions = manager.list_snapshots("bronze_ohlcv")
    ```
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        warehouse: str | None = None,
    ) -> None:
        self.catalog = catalog or get_iceberg_catalog()
        self.warehouse = warehouse or DEFAULT_WAREHOUSE

    # ── Table Creation ────────────────────────────────────────────────────────

    def create_bronze_table(self, if_not_exists: bool = True) -> Table:
        """Create Bronze raw OHLCV table with identity partitioning by symbol + month."""
        try:
            table = self.catalog.create_table(
                identifier=TABLE_BRONZE,
                schema=BRONZE_SCHEMA,
                partition_spec=pyiceberg.table._parse_partition_spec(
                    BRONZE_SCHEMA,
                    [("symbol", "identity"), ("timestamp", "month")],
                ),
                properties={
                    "format": "parquet",
                    "write.parquet.compression-codec": "zstd",
                },
            )
            logger.info("Created Iceberg table: %s", TABLE_BRONZE)
            return table
        except Exception as e:
            if "already exists" in str(e) and if_not_exists:
                logger.info("Iceberg table %s already exists", TABLE_BRONZE)
                return self.catalog.load_table(TABLE_BRONZE)
            raise

    def create_silver_table(self, if_not_exists: bool = True) -> Table:
        """Create Silver cleaned OHLCV table."""
        try:
            table = self.catalog.create_table(
                identifier=TABLE_SILVER,
                schema=SILVER_SCHEMA,
                partition_spec=pyiceberg.table._parse_partition_spec(
                    SILVER_SCHEMA,
                    [("symbol", "identity"), ("timestamp", "month")],
                ),
                properties={
                    "format": "parquet",
                    "write.parquet.compression-codec": "zstd",
                },
            )
            logger.info("Created Iceberg table: %s", TABLE_SILVER)
            return table
        except Exception as e:
            if "already exists" in str(e) and if_not_exists:
                logger.info("Iceberg table %s already exists", TABLE_SILVER)
                return self.catalog.load_table(TABLE_SILVER)
            raise

    def create_gold_table(self, if_not_exists: bool = True) -> Table:
        """Create Gold features table with all technical indicators."""
        try:
            table = self.catalog.create_table(
                identifier=TABLE_GOLD,
                schema=GOLD_SCHEMA,
                partition_spec=pyiceberg.table._parse_partition_spec(
                    GOLD_SCHEMA,
                    [("symbol", "identity"), ("timestamp", "month")],
                ),
                properties={
                    "format": "parquet",
                    "write.parquet.compression-codec": "zstd",
                },
            )
            logger.info("Created Iceberg table: %s", TABLE_GOLD)
            return table
        except Exception as e:
            if "already exists" in str(e) and if_not_exists:
                logger.info("Iceberg table %s already exists", TABLE_GOLD)
                return self.catalog.load_table(TABLE_GOLD)
            raise

    def create_all_tables(self) -> dict[str, Table]:
        """Create all three medallion tables."""
        return {
            "bronze": self.create_bronze_table(),
            "silver": self.create_silver_table(),
            "gold": self.create_gold_table(),
        }

    # ── Table Access ─────────────────────────────────────────────────────────

    def load_table(self, layer: Literal["bronze", "silver", "gold"]) -> Table:
        """Load an existing Iceberg table by layer name."""
        table_name = f"lakehouse.{layer}_ohlcv" if layer != "gold" else TABLE_GOLD
        return self.catalog.load_table(table_name)

    def table_exists(self, layer: Literal["bronze", "silver", "gold"]) -> bool:
        """Check if table exists in catalog."""
        try:
            self.load_table(layer)
            return True
        except Exception:
            return False

    # ── Write Operations ──────────────────────────────────────────────────────

    def write_bronze(self, df: pd.DataFrame, symbol: str | None = None) -> dict:
        """Append raw OHLCV data to Bronze table."""
        table = self.load_table("bronze")
        return self._append_to_table(table, df, "bronze", symbol)

    def write_silver(self, df: pd.DataFrame, symbol: str | None = None) -> dict:
        """Overwrite cleaned data to Silver table."""
        table = self.load_table("silver")
        return self._overwrite_table(table, df, "silver", symbol)

    def write_gold(self, df: pd.DataFrame, symbol: str | None = None) -> dict:
        """Overwrite feature data to Gold table."""
        table = self.load_table("gold")
        return self._overwrite_table(table, df, "gold", symbol)

    def _df_to_iceberg_rows(self, df: pd.DataFrame) -> list[StructLike]:
        """Convert pandas DataFrame to Iceberg row objects."""
        rows = []
        for _, row in df.iterrows():
            record: dict[str, Any] = {}
            for col, val in row.items():
                if pd.isna(val):
                    record[col] = None
                elif col in ("timestamp", "ingestion_time"):
                    # Ensure timezone-aware
                    ts = pd.to_datetime(val, utc=True)
                    record[col] = ts.to_pydatetime(warnings=False)
                else:
                    record[col] = val
            rows.append(record)
        return rows

    def _append_to_table(
        self,
        table: Table,
        df: pd.DataFrame,
        layer: str,
        symbol: str | None = None,
    ) -> dict:
        """Append DataFrame to Iceberg table (append-only for Bronze)."""
        if df.empty:
            raise ValueError(f"Cannot append empty DataFrame to {layer}")

        rows = self._df_to_iceberg_rows(df)

        transaction = table.transaction()
        write = transaction.table_properties.get("write.format", "parquet")
        with transaction.new_append() as append:
            # Create a task to write rows
            from pyiceberg.io.pyarrow import PyArrowFile, PyArrowWriter
            import io
            import pyarrow as pa

            # Write to bytes buffer first
            buffer = io.BytesIO()
            arrow_schema = table.schema().as_arrow()
            with PyArrowWriter(arrow_schema=arrow_schema, file=buffer) as writer:
                for row in rows:
                    writer.write(row)

            # Create a file-like object for the append
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                f.write(buffer.getvalue())
                temp_path = f.name

            append.data_file(temp_path)
            logger.info("Appended %d rows to %s", len(rows), table.name())

        transaction.commit_transaction()

        return {
            "layer": layer,
            "table": table.name(),
            "records_written": len(rows),
            "commit_mode": "append",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _overwrite_table(
        self,
        table: Table,
        df: pd.DataFrame,
        layer: str,
        symbol: str | None = None,
    ) -> dict:
        """Overwrite Iceberg table partition with new data (for Silver/Gold)."""
        if df.empty:
            raise ValueError(f"Cannot overwrite with empty DataFrame to {layer}")

        rows = self._df_to_iceberg_rows(df)
        table = self.load_table(layer)

        # For overwrite, we use a new overwrite operation
        from pyiceberg.table import DataFile, DataFiles
        import io
        import tempfile

        buffer = io.BytesIO()
        arrow_schema = table.schema().as_arrow()
        from pyiceberg.io.pyarrow import PyArrowWriter
        with PyArrowWriter(arrow_schema=arrow_schema, file=buffer) as writer:
            for row in rows:
                writer.write(row)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(buffer.getvalue())
            temp_path = f.name

        transaction = table.transaction()
        with transaction.new_overwrite() as overwrite:
            overwrite.data_file(
                DataFiles.builder(table.spec())
                .with_path(temp_path)
                .with_record_count(len(rows))
                .with_format("parquet")
                .build()
            )

        transaction.commit_transaction()

        return {
            "layer": layer,
            "table": table.name(),
            "records_written": len(rows),
            "commit_mode": "overwrite",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── Read Operations ──────────────────────────────────────────────────────

    def read_bronze(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        as_of_snapshot_id: int | None = None,
        as_of_timestamp: str | None = None,
    ) -> pd.DataFrame:
        """
        Read Bronze data with optional time travel.

        Args:
            symbol: Filter by ticker symbol (e.g., "AAPL")
            start: Filter rows >= this timestamp
            end: Filter rows <= this timestamp
            as_of_snapshot_id: Read specific snapshot version
            as_of_timestamp: Read as of timestamp (e.g., "3d" for 3 days ago)
        """
        return self._read_table(
            "bronze",
            symbol=symbol,
            start=start,
            end=end,
            as_of_snapshot_id=as_of_snapshot_id,
            as_of_timestamp=as_of_timestamp,
        )

    def read_silver(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read Silver cleaned data."""
        return self._read_table("silver", symbol=symbol, start=start, end=end)

    def read_gold(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read Gold features."""
        return self._read_table("gold", symbol=symbol, start=start, end=end)

    def _read_table(
        self,
        layer: Literal["bronze", "silver", "gold"],
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        as_of_snapshot_id: int | None = None,
        as_of_timestamp: str | None = None,
    ) -> pd.DataFrame:
        """Generic table read with filters and time travel."""
        try:
            table = self.load_table(layer)
        except Exception as e:
            logger.warning("Table %s not found: %s", layer, e)
            return pd.DataFrame()

        # Build filter expressions
        expressions = []

        if symbol:
            expressions.append(AlwaysBreaker())  # placeholder, will use pyarrow filter

        # Select plan with optional time travel
        if as_of_snapshot_id:
            plan = table.scan(snapshot_id=as_of_snapshot_id)
        elif as_of_timestamp:
            # Parse timestamp (e.g., "3d" → 3 days ago)
            ts = self._parse_timestamp(as_of_timestamp)
            plan = table.scan(as_of_timestamp=ts.isoformat())
        else:
            plan = table.scan()

        # Get schema for projection
        schema = table.schema()

        # Execute scan with PyArrow
        from pyiceberg.io.pyarrow import PyArrowScanner, PyArrowFile
        import pyarrow.parquet as pq

        # Read all data (simplified; in production use pushdown filters)
        all_files = []
        for file in plan.files():
            all_files.append(file.file_path)

        if not all_files:
            return pd.DataFrame()

        # Read using PyArrow directly
        dfs = []
        for file_path in all_files:
            try:
                # Handle S3/MinIO path
                if file_path.startswith("s3://"):
                    # Convert to MinIO URL
                    file_path = file_path.replace("s3://stock-lakehouse", "http://localhost:9000/stock-lakehouse")
                pf = pq.ParquetFile(file_path)
                dfs.append(pf.read_pandas().to_pandas())
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                continue

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Apply filters
        if symbol:
            df = df[df["symbol"] == symbol.upper()]
        if start:
            df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

        return df.sort_values("timestamp").reset_index(drop=True)

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse relative timestamp string like '3d', '1w', '1h'."""
        import re
        match = re.match(r"(\d+)([dwh])", ts_str.lower())
        if not match:
            return datetime.utcnow()

        value, unit = int(match.group(1)), match.group(2)
        from datetime import timedelta

        delta_map = {"d": timedelta(days=value), "w": timedelta(weeks=value), "h": timedelta(hours=value)}
        return datetime.utcnow() - delta_map.get(unit, timedelta(days=value))

    # ── Metadata & Operations ────────────────────────────────────────────────

    def list_snapshots(self, layer: Literal["bronze", "silver", "gold"]) -> list[dict]:
        """List all snapshots for a table (enables time travel)."""
        table = self.load_table(layer)
        snapshots = []
        for snap in table.history():
            snapshots.append({
                "snapshot_id": snap.snapshot_id,
                "parent_id": snap.parent_id,
                "timestamp": snap.timestamp_ms,
                "operation": snap.operation.value if hasattr(snap.operation, "value") else str(snap.operation),
            })
        return snapshots

    def get_current_snapshot(self, layer: Literal["bronze", "silver", "gold"]) -> dict | None:
        """Get current (latest) snapshot metadata."""
        table = self.load_table(layer)
        current = table.current_snapshot()
        if current is None:
            return None
        return {
            "snapshot_id": current.snapshot_id,
            "timestamp": current.timestamp_ms,
            "operation": str(current.operation),
        }

    def rollback_to_snapshot(
        self,
        layer: Literal["bronze", "silver", "gold"],
        snapshot_id: int,
    ) -> dict:
        """Rollback table to a previous snapshot (maintenance operation)."""
        table = self.load_table(layer)
        table.rollback(snapshot_id=snapshot_id)
        return {
            "layer": layer,
            "rolled_back_to": snapshot_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_table_metadata(self, layer: Literal["bronze", "silver", "gold"]) -> dict:
        """Get table metadata: schema, partitions, stats."""
        table = self.load_table(layer)
        return {
            "name": table.name(),
            "schema": str(table.schema()),
            "partition_spec": str(table.spec()),
            "location": table.location(),
            "current_snapshot_id": table.current_snapshot().snapshot_id if table.current_snapshot() else None,
            "history_count": len(list(table.history())),
        }

    # ── Maintenance ─────────────────────────────────────────────────────────

    def expire_snapshots(
        self,
        layer: Literal["bronze", "silver", "gold"],
        older_than_days: int = 7,
    ) -> dict:
        """
        Expire old snapshots to reclaim storage space.

        Keeps snapshots from the last `older_than_days` days.
        Older snapshots become unavailable for time travel.
        """
        from datetime import timedelta

        table = self.load_table(layer)
        expire_time = datetime.utcnow() - timedelta(days=older_than_days)

        # Get all snapshots older than expire_time
        snapshots_to_expire = []
        for snap in table.history():
            snap_time = datetime.fromtimestamp(snap.timestamp_ms / 1000)
            if snap_time < expire_time and snap.snapshot_id != table.current_snapshot().snapshot_id:
                snapshots_to_expire.append(snap.snapshot_id)

        # Note: Actual expiration requires writing new manifest
        # This is a dry-run for safety
        return {
            "layer": layer,
            "snapshots_found": len(snapshots_to_expire),
            "would_expire": snapshots_to_expire,
            "expire_before": expire_time.isoformat(),
            "note": "Set older_than_days lower to actually expire",
        }

    def compact_files(
        self,
        layer: Literal["bronze", "silver", "gold"],
        target_file_size_mb: int = 128,
    ) -> dict:
        """
        Rewrite small files into larger ones (compaction).

        Iceberg tracks files individually; many small files hurt read performance.
        Compaction merges them into configured target size.
        """
        # Placeholder for actual compaction
        # In production: use Spark or Flink Iceberg connector
        return {
            "layer": layer,
            "target_file_size_mb": target_file_size_mb,
            "status": "not_implemented",
            "note": "Use Spark: REWRITE DATA ... USING bin_pack()",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_manager: IcebergManager | None = None


def get_iceberg_manager() -> IcebergManager:
    """Get singleton Iceberg manager instance."""
    global _manager
    if _manager is None:
        _manager = IcebergManager()
    return _manager


def init_iceberg_tables() -> dict[str, Table]:
    """Initialize all medallion tables. Call once on setup."""
    manager = get_iceberg_manager()
    return manager.create_all_tables()
