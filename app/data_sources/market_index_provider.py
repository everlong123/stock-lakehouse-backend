"""Market Index provider - fetches VN-Index, HNX, UPCOM and sector indices."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class MarketIndexProvider:
    """Fetch Vietnam market indices: VN-Index, HNX, UPCOM, and sector indices."""

    source_name = "market_index"

    # Index symbols
    INDEX_SYMBOLS = {
        "VNINDEX": "^VNINDEX",
        "HNX": "^HNX",
        "UPCOM": "^UPCOM",
        "VN30": "^VN30",
        "HNX30": "^HNX30",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
        })

    def get_index_data(self, symbol: str = "VNINDEX") -> pd.DataFrame:
        """Get historical data for a market index."""
        # Map to Yahoo Finance symbol
        yahoo_symbol = self.INDEX_SYMBOLS.get(symbol.upper(), symbol.upper())

        try:
            # Use Yahoo Finance
            response = self.session.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
                params={"interval": "1d", "range": "2y"},
                timeout=15,
            )

            if response.status_code != 200:
                logger.warning(f"Yahoo Finance returned {response.status_code} for {symbol}")
                return pd.DataFrame()

            data = response.json()
            result = data.get("chart", {}).get("result", [{}])[0]

            if not result:
                return pd.DataFrame()

            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps:
                return pd.DataFrame()

            records = []
            for i, ts in enumerate(timestamps):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                close = quotes.get("close", [None])[i]
                high = quotes.get("high", [None])[i]
                low = quotes.get("low", [None])[i]
                open_price = quotes.get("open", [None])[i]
                volume = quotes.get("volume", [None])[i]

                if close:
                    records.append({
                        "symbol": symbol.upper(),
                        "timestamp": dt,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "source": self.source_name,
                        "ingestion_time": pd.Timestamp.now(tz="UTC"),
                    })

            frame = pd.DataFrame(records)
            if not frame.empty:
                frame["change"] = frame["close"].pct_change() * 100
                frame["change_from_open"] = ((frame["close"] - frame["open"]) / frame["open"]) * 100
            return frame

        except Exception as exc:
            logger.warning(f"Failed to fetch {symbol}: {exc}")
            return pd.DataFrame()

    def get_all_indices(self) -> dict[str, pd.DataFrame]:
        """Get all major indices."""
        results = {}
        for index_name in self.INDEX_SYMBOLS.keys():
            results[index_name] = self.get_index_data(index_name)
        return results

    def get_index_summary(self, symbol: str = "VNINDEX") -> dict[str, Any]:
        """Get current summary for an index."""
        df = self.get_index_data(symbol)

        if df.empty:
            return {"symbol": symbol, "error": "No data available"}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        return {
            "symbol": symbol,
            "close": float(latest["close"]),
            "change": float(latest.get("change", 0)),
            "change_pct": float(latest.get("change", 0)),
            "volume": int(latest["volume"]) if pd.notna(latest["volume"]) else 0,
            "high_52w": float(df["high"].max()) if not df.empty else None,
            "low_52w": float(df["low"].min()) if not df.empty else None,
            "prev_close": float(prev["close"]),
            "timestamp": str(latest["timestamp"]),
        }

    def get_marketBreadth(self, date: str | None = None) -> dict[str, Any]:
        """Get market breadth (advance/decline, new high/low)."""
        # This would ideally come from a real-time data provider
        # For now, return placeholder structure
        return {
            "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "new_highs": 0,
            "new_lows": 0,
            "total_traded": 0,
            "market_sentiment": "neutral",
        }

    def get_sector_performance(self) -> pd.DataFrame:
        """Get sector performance data."""
        # VN sector indices
        sector_map = {
            "VNALL": "VNAll",
            "VNMID": "VNMid",
            "VNSML": "VNSmall",
            "VNFIN": "VNFin",
            "VNIND": "VNInd",
            "VNCON": "VNCon",
            "VNREAL": "VNReal",
            "VNUTI": "VNUtil",
            "VNHEAL": "VNHealth",
            "VNIT": "VNIT",
        }

        records = []
        for yahoo_sym, name in sector_map.items():
            try:
                df = self.get_index_data(name)
                if not df.empty:
                    latest = df.iloc[-1]
                    year_change = (latest["close"] / df.iloc[0]["close"] - 1) * 100 if len(df) > 1 else 0

                    records.append({
                        "sector": name,
                        "close": latest["close"],
                        "change_pct": latest.get("change", 0),
                        "year_change_pct": year_change,
                        "volume": latest["volume"] if pd.notna(latest["volume"]) else 0,
                    })
            except Exception:
                continue

        return pd.DataFrame(records)
