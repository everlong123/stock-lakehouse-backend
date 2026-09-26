"""Parquet partition helpers for the medallion layers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def partition_relative_path(symbol: str, timestamp: datetime, filename: str = "part-001.parquet") -> str:
    """Build a Hive-style partition path: symbol=AAPL/year=2026/month=09/part-001.parquet."""
    year = int(timestamp.year)
    month = int(timestamp.month)
    return f"symbol={symbol.upper()}/year={year}/month={month:02d}/{filename}"


def add_partition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add year/month partition columns from timestamp."""
    result = frame.copy()
    ts = pd.to_datetime(result["timestamp"], utc=True)
    result["year"] = ts.dt.year.astype(int)
    result["month"] = ts.dt.month.astype(int)
    return result


def split_by_partition(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group rows by symbol/year/month partition path."""
    partitioned = add_partition_columns(frame)
    grouped: dict[str, pd.DataFrame] = {}
    for (symbol, year, month), group in partitioned.groupby(["symbol", "year", "month"], sort=True):
        path = f"symbol={symbol}/year={int(year)}/month={int(month):02d}/part-001.parquet"
        grouped[path] = group.reset_index(drop=True)
    return grouped
