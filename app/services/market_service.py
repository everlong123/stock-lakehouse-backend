"""Market data service reading from Gold/Silver/Bronze."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.core.exceptions import NotFoundError
from app.lakehouse.bronze import BronzeLayer
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer


class MarketService:
    """Read OHLCV from the lakehouse, preferring Gold then Silver then Bronze."""

    def get_history(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        frame = GoldLayer().read(symbol)
        source_layer = "gold"
        if frame.empty:
            frame = SilverLayer().read(symbol)
            source_layer = "silver"
        if frame.empty:
            frame = BronzeLayer().read(symbol)
            source_layer = "bronze"
        if frame.empty:
            raise NotFoundError(f"No market data available for {symbol} in the selected range.")
        frame = frame.sort_values("timestamp")
        if start:
            frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
        if frame.empty:
            raise NotFoundError("No market data available for the selected range.")
        frame = frame.copy()
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame["layer"] = source_layer
        frame["interval"] = interval
        return frame.reset_index(drop=True)

    def get_latest(self, symbol: str, interval: str = "1d") -> dict:
        frame = self.get_history(symbol, interval=interval)
        last = frame.iloc[-1]
        prev_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else float(last["close"])
        change_pct = (float(last["close"]) - prev_close) / prev_close if prev_close else 0.0
        return {
            "symbol": symbol.upper(),
            "timestamp": str(last["timestamp"]),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "adj_close": float(last.get("adj_close", last["close"])),
            "volume": float(last["volume"]),
            "change_pct": change_pct,
            "source_layer": last.get("layer", "gold"),
            "interval": interval,
        }

    def to_records(self, frame: pd.DataFrame, limit: int = 1500) -> list[dict]:
        subset = frame.tail(limit)
        records = []
        for row in subset.itertuples(index=False):
            records.append(
                {
                    "symbol": str(getattr(row, "symbol", "")),
                    "timestamp": str(row.timestamp),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "adj_close": float(getattr(row, "adj_close", row.close)),
                    "volume": float(row.volume),
                    "source": str(getattr(row, "source", "lakehouse")),
                }
            )
        return records
