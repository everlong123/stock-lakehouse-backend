"""Silver layer for news and sentiment data."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.core.logging_config import get_logger
from app.lakehouse.parquet_manager import split_by_partition
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)


NEWS_SCHEMA = {
    "symbol": "string",
    "title": "string",
    "content": "string",
    "url": "string",
    "published_date": "datetime64[ns, UTC]",
    "source": "string",
    "ingestion_time": "datetime64[ns, UTC]",
}


class NewsSilverLayer:
    """Cleaned news/sentiment zone."""

    layer_name = "silver_news"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def append(self, frame: pd.DataFrame, lineage: dict | None = None) -> dict:
        """Append cleaned news records."""
        if frame.empty:
            return {"records_written": 0}

        # Validate schema
        clean = frame.copy()
        clean["symbol"] = clean["symbol"].fillna("MARKET").str.upper()
        clean["title"] = clean["title"].fillna("").str.strip()
        clean["content"] = clean["content"].fillna("").str.strip()
        clean["url"] = clean["url"].fillna("")
        clean["published_date"] = pd.to_datetime(clean["published_date"], utc=True)
        clean["ingestion_time"] = pd.to_datetime(clean.get("ingestion_time", pd.Timestamp.now(tz="UTC")), utc=True)

        # Add derived fields
        clean["title_length"] = clean["title"].str.len()
        clean["has_content"] = clean["content"].str.len() > 0

        # Partition by date for efficient querying
        clean["date_partition"] = clean["published_date"].dt.strftime("%Y-%m-%d")

        # Write by date partition
        partitions = {}
        for date, group in clean.groupby("date_partition"):
            partitions[f"date={date}"] = group.drop(columns=["date_partition"])

        written_paths = []
        for relative_path, group in partitions.items():
            path = self.storage.write_parquet(self.layer_name, relative_path, group)
            written_paths.append(path)

        metadata = {
            "layer": self.layer_name,
            "records_written": int(len(clean)),
            "symbols": clean["symbol"].unique().tolist(),
            "date_range": {
                "min": str(clean["published_date"].min()),
                "max": str(clean["published_date"].max()),
            },
            "lineage": lineage or {},
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"News Silver append: {len(clean)} records, {len(written_paths)} partitions")
        return metadata

    def read_by_symbol(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Read news for a symbol within days."""
        symbol = symbol.upper()
        frame = self.storage.read_prefix(self.layer_name, "date=")

        if frame.empty:
            return frame

        # Filter by symbol
        frame = frame[frame["symbol"] == symbol]

        # Filter by date
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame[frame["published_date"] >= cutoff]

        return frame.sort_values("published_date", ascending=False).reset_index(drop=True)

    def read_market_news(self, days: int = 7) -> pd.DataFrame:
        """Read general market news (symbol = MARKET)."""
        frame = self.storage.read_prefix(self.layer_name, "date=")

        if frame.empty:
            return frame

        frame = frame[frame["symbol"] == "MARKET"]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame[frame["published_date"] >= cutoff]

        return frame.sort_values("published_date", ascending=False).reset_index(drop=True)
