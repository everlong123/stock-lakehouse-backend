"""Vietnamese stock data provider - fetches from CafeF, VietStock, and financial news sites."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS
from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider

logger = get_logger(__name__)


class VNStockProvider(StockDataProvider):
    """Fetch Vietnamese stock data from CafeF API (no API key required)."""
    
    source_name = "vn_stock"
    
    # VN Index components
    VN30_SYMBOLS = [
        "VNM", "VIC", "VPB", "TCB", "MBB", "SSI", "ACB", "CTG", "MWG", "FPT",
        "PNJ", "HDB", "STB", "TPB", "MSB", "SHB", "LPB", "EIB", "OCB", "VCB",
        "BID", "gas", "vnm", "vhm", "vre", "sab", "imp", "pla", "pdr", "nd2"
    ]
    
    # CafeF API endpoints (free)
    CAFEF_API = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
            "Referer": "https://s.cafef.vn/",
        })
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists on Vietnamese exchanges."""
        symbol = symbol.upper()
        # VN stocks are typically 3-4 letters (e.g., VNM, FPT, VCB)
        if len(symbol) <= 5 and symbol.isalpha():
            return True
        return False
    
    def get_historical_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV from CafeF."""
        symbol = symbol.upper()
        end = end or datetime.now()
        start = start or datetime(end.year - 2, end.month, end.day)
        
        records = self._fetch_cafef_api(symbol, start, end, interval)
        
        if not records:
            raise DataSourceError(f"No data found for {symbol}")
        
        frame = pd.DataFrame(records)
        frame["symbol"] = symbol
        frame["source"] = self.source_name
        frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        # Ensure correct column order
        for col in OHLCV_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        
        frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        
        # Filter by date range
        if start:
            frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
        if end:
            frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
        
        logger.info("Fetched %s rows for VN stock %s", len(frame), symbol)
        return frame
    
    def _fetch_cafef_api(self, symbol: str, start: datetime, end: datetime, interval: str) -> list:
        """Fetch from CafeF API."""
        # Map interval
        page_index = 1
        if interval == "1d":
            page_index = 1
        elif interval == "1w":
            page_index = 2
        elif interval == "1M":
            page_index = 3
        
        records = []
        current_page = 1
        max_pages = 20  # Limit to prevent infinite loop
        
        while current_page <= max_pages:
            params = {
                "Symbol": symbol,
                "StartDate": start.strftime("%d/%m/%Y"),
                "EndDate": end.strftime("%d/%m/%Y"),
                "PageIndex": current_page,
                "PageSize": 200,
            }
            
            try:
                response = self.session.get(self.CAFEF_API, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if not data or "data" not in data or not data["data"]:
                    break
                
                for item in data["data"]:
                    try:
                        dt = datetime.strptime(item["Date"], "%d/%m/%Y %H:%M:%S")
                        dt = dt.replace(tzinfo=timezone.utc)
                        
                        records.append({
                            "timestamp": dt,
                            "open": float(item["Open"].replace(",", "")) if item.get("Open") else 0,
                            "high": float(item["High"].replace(",", "")) if item.get("High") else 0,
                            "low": float(item["Low"].replace(",", "")) if item.get("Low") else 0,
                            "close": float(item["Close"].replace(",", "")) if item.get("Close") else 0,
                            "adj_close": float(item["AdjustPrice"].replace(",", "")) if item.get("AdjustPrice") else 0,
                            "volume": float(item["Volume"].replace(",", "")) if item.get("Volume") else 0,
                        })
                    except (ValueError, KeyError):
                        continue
                
                if len(data["data"]) < 200:
                    break
                current_page += 1
                
            except requests.exceptions.RequestException as exc:
                logger.warning("CafeF API error for %s: %s", symbol, exc)
                break
            except Exception as exc:
                logger.warning("Parse error for %s: %s", symbol, exc)
                break
        
        return records
    
    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """Get most recent 5 days of data."""
        end = datetime.now()
        start = end - pd.Timedelta(days=30)
        frame = self.get_historical_data(symbol, start, end, interval)
        return frame.tail(5).reset_index(drop=True)
    
    def get_index_components(self, index: Literal["VN-Index", "VN30", "HNX", "UPCOM"]) -> list[str]:
        """Get list of symbols for a Vietnamese index."""
        # VN30 components (approximate - based on typical composition)
        vn30 = ["VNM", "VIC", "VPB", "TCB", "MBB", "SSI", "ACB", "CTG", "MWG", "FPT",
                "PNJ", "HDB", "STB", "TPB", "MSB", "SHB", "LPB", "EIB", "OCB", "VCB",
                "BID", "GAS", "VHM", "VRE", "SAB", "IMP", "PLD", "PDR", "NVL", "BCM"]
        
        hnx = ["AAV", "ADS", "ART", "BII", "CAN", "CLC", "CMS", "DBC", "DDG", "DHC",
               "DTD", "GMC", "HUT", "IDC", "KLS", "L14", "LAS", "LHC", "MBS", "MDC"]
        
        if index == "VN30":
            return vn30
        elif index == "HNX":
            return hnx
        elif index == "VN-Index":
            return vn30 + hnx
        return []
