"""Silver layer for fundamental financial data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.logging_config import get_logger
from app.lakehouse.storage_base import StorageBackend
from app.lakehouse.storage_factory import get_storage_backend

logger = get_logger(__name__)


class FundamentalsSilverLayer:
    """Cleaned fundamental financial data zone."""

    layer_name = "silver_fundamentals"

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self.storage = storage or get_storage_backend()

    def append_metrics(self, metrics: dict[str, Any], lineage: dict | None = None) -> dict:
        """Append fundamental metrics for a symbol."""
        # Convert to single-row DataFrame
        record = {
            "symbol": metrics.get("symbol", "").upper(),
            "timestamp": pd.Timestamp.now(tz="UTC"),
            # Valuation
            "market_cap": metrics.get("market_cap"),
            "pe_ratio": metrics.get("pe_ratio"),
            "pb_ratio": metrics.get("pb_ratio"),
            "ps_ratio": metrics.get("ps_ratio"),
            "peg_ratio": metrics.get("peg_ratio"),
            "ev_ebitda": metrics.get("ev_ebitda"),
            "dividend_yield": metrics.get("dividend_yield"),
            # Profitability
            "roe": metrics.get("roe"),
            "roa": metrics.get("roa"),
            "roic": metrics.get("roic"),
            "gross_margin": metrics.get("gross_margin"),
            "operating_margin": metrics.get("operating_margin"),
            "net_margin": metrics.get("net_margin"),
            # Growth
            "revenue_growth_yoy": metrics.get("revenue_growth_yoy"),
            "profit_growth_yoy": metrics.get("profit_growth_yoy"),
            "eps_growth_yoy": metrics.get("eps_growth_yoy"),
            # Financial Health
            "debt_equity": metrics.get("debt_equity"),
            "current_ratio": metrics.get("current_ratio"),
            "quick_ratio": metrics.get("quick_ratio"),
            # Per Share
            "eps": metrics.get("eps"),
            "bvps": metrics.get("bvps"),
            "dps": metrics.get("dps"),
        }

        df = pd.DataFrame([record])
        return self.append(df, lineage)

    def append(self, frame: pd.DataFrame, lineage: dict | None = None) -> dict:
        """Append fundamental data records."""
        if frame.empty:
            return {"records_written": 0}

        clean = frame.copy()

        # Ensure symbol is uppercase
        if "symbol" in clean.columns:
            clean["symbol"] = clean["symbol"].str.upper()

        # Ensure timestamp
        if "timestamp" not in clean.columns:
            clean["timestamp"] = pd.Timestamp.now(tz="UTC")
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)

        # Parse period from income statement data
        if "period" in clean.columns:
            clean["period"] = pd.to_datetime(clean["period"], utc=True)
            clean["year"] = clean["period"].dt.year
            clean["quarter"] = clean["period"].dt.quarter

        # Write by symbol partition
        partition_key = f"symbol={clean['symbol'].iloc[0]}"
        path = self.storage.write_parquet(self.layer_name, partition_key, clean)

        metadata = {
            "layer": self.layer_name,
            "records_written": int(len(clean)),
            "symbols": clean["symbol"].unique().tolist(),
            "path": path,
            "lineage": lineage or {},
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Fundamentals Silver append: {len(clean)} records for {clean['symbol'].iloc[0]}")
        return metadata

    def read_latest(self, symbol: str) -> pd.DataFrame:
        """Read latest fundamental data for a symbol."""
        symbol = symbol.upper()
        frame = self.storage.read_prefix(self.layer_name, f"symbol={symbol}")

        if frame.empty:
            return frame

        # Get most recent
        return frame.sort_values("timestamp", ascending=False).head(1)

    def read_history(self, symbol: str, years: int = 4) -> pd.DataFrame:
        """Read fundamental history for a symbol."""
        symbol = symbol.upper()
        frame = self.storage.read_prefix(self.layer_name, f"symbol={symbol}")

        if frame.empty:
            return frame

        # Filter by years
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years)
        frame = frame[frame["timestamp"] >= cutoff]

        return frame.sort_values("timestamp", ascending=False)
