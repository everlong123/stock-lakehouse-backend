"""Gold layer for aggregated features from multiple sources."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.logging_config import get_logger
from app.lakehouse.parquet_manager import split_by_partition
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)


class GoldFeaturesLayer:
    """
    Aggregated feature layer that combines:
    - Sentiment aggregates (daily, weekly)
    - Macro features (interest rates, CPI, FX)
    - Market regime indicators
    """

    layer_name = "gold_features"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def write_sentiment_agg(self, df: pd.DataFrame) -> dict:
        """Write daily sentiment aggregates."""
        if df.empty:
            return {"records_written": 0}

        # Ensure required columns
        df = df.copy()
        df["date"] = pd.to_datetime(df["published_date"], utc=True).dt.date
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Aggregate by symbol and date
        agg = df.groupby(["symbol", "date"]).agg({
            "sentiment": ["mean", "std", "min", "max"],
            "sentiment_confidence": "mean",
            "title": "count",
        }).reset_index()

        # Flatten column names
        agg.columns = [
            "symbol", "date",
            "sentiment_mean", "sentiment_std", "sentiment_min", "sentiment_max",
            "sentiment_confidence_mean",
            "article_count",
        ]

        # Add derived features
        agg["sentiment_positive"] = (agg["sentiment_mean"] > 0.2).astype(int)
        agg["sentiment_negative"] = (agg["sentiment_mean"] < -0.2).astype(int)
        agg["sentiment_range"] = agg["sentiment_max"] - agg["sentiment_min"]

        # Add timestamp for compatibility
        agg["timestamp"] = agg["date"]
        agg["source"] = "sentiment_agg"

        # Write
        partition_key = "type=sentiment"
        path = self.storage.write_parquet(self.layer_name, partition_key, agg)

        logger.info(f"Sentiment agg written: {len(agg)} records")
        return {"records_written": len(agg), "path": path}

    def write_macro_features(self, macro_data: dict[str, Any]) -> dict:
        """Write macro features snapshot."""
        record = {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "source": "macro",
        }

        # Exchange rate
        if "exchange_rate" in macro_data:
            record["usd_vnd"] = macro_data["exchange_rate"].get("usd_vnd")
            record["fx_timestamp"] = macro_data["exchange_rate"].get("timestamp")

        # Gold
        if "gold" in macro_data:
            record["gold_usd_oz"] = macro_data["gold"].get("price_usd_oz")
            record["gold_timestamp"] = macro_data["gold"].get("timestamp")

        # Oil
        if "oil" in macro_data:
            record["wti_usd_barrel"] = macro_data["oil"].get("wti_usd_barrel")

        df = pd.DataFrame([record])

        partition_key = "type=macro"
        path = self.storage.write_parquet(self.layer_name, partition_key, df)

        return {"records_written": len(df), "path": path}

    def write_index_features(self, index_data: dict[str, Any]) -> dict:
        """Write market index features."""
        records = []

        for index_name, data in index_data.items():
            if isinstance(data, dict):
                record = {
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "index_name": index_name,
                    "source": "market_index",
                    "close": data.get("close"),
                    "change_pct": data.get("change_pct"),
                    "volume": data.get("volume"),
                    "high_52w": data.get("high_52w"),
                    "low_52w": data.get("low_52w"),
                }
                records.append(record)

        if not records:
            return {"records_written": 0}

        df = pd.DataFrame(records)

        partition_key = "type=market_index"
        path = self.storage.write_parquet(self.layer_name, partition_key, df)

        return {"records_written": len(df), "path": path}

    def read_sentiment_agg(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Read aggregated sentiment for a symbol."""
        frame = self.storage.read_prefix(self.layer_name, "type=sentiment")

        if frame.empty:
            return frame

        frame = frame[frame["symbol"] == symbol.upper()]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame[frame["timestamp"] >= cutoff]

        return frame.sort_values("timestamp").reset_index(drop=True)

    def read_macro(self, days: int = 30) -> pd.DataFrame:
        """Read macro features history."""
        frame = self.storage.read_prefix(self.layer_name, "type=macro")

        if frame.empty:
            return frame

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame[frame["timestamp"] >= cutoff]

        return frame.sort_values("timestamp").reset_index(drop=True)

    def read_index_features(self, index_name: str | None = None, days: int = 30) -> pd.DataFrame:
        """Read market index features."""
        frame = self.storage.read_prefix(self.layer_name, "type=market_index")

        if frame.empty:
            return frame

        if index_name:
            frame = frame[frame["index_name"] == index_name]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame[frame["timestamp"] >= cutoff]

        return frame.sort_values("timestamp").reset_index(drop=True)


class GoldSentimentFeatures:
    """Build sentiment-derived features for ML models."""

    @staticmethod
    def build_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
        """Build features from news data for ML."""
        if news_df.empty:
            return pd.DataFrame()

        df = news_df.copy()
        df["date"] = pd.to_datetime(df["published_date"], utc=True).dt.date

        # Aggregate daily features
        daily = df.groupby(["symbol", "date"]).agg({
            "sentiment": ["mean", "std", "count"],
            "sentiment_confidence": "mean",
            "sentiment_label": lambda x: (x == "positive").sum(),
        }).reset_index()

        daily.columns = [
            "symbol", "date",
            "sentiment_mean", "sentiment_std", "article_count",
            "sentiment_confidence", "positive_count"
        ]

        # Rolling features
        for window in [3, 7, 14]:
            daily[f"sentiment_ma_{window}"] = daily.groupby("symbol")["sentiment_mean"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            daily[f"sentiment_count_ma_{window}"] = daily.groupby("symbol")["article_count"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )

        return daily
