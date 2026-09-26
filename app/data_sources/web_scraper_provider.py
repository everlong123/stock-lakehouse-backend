"""Web scraper provider for stock data - no API key required."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.constants import OHLCV_COLUMNS
from app.core.exceptions import DataSourceError, DataValidationError
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider

logger = get_logger(__name__)

# Vietnamese stock suffix mapping for Yahoo Finance
VN_SYMBOLS = {
    "VCB", "TCB", "MBB", "ACB", "BID", "SSI", "VND", "VHM", "VRE", "KDH",
    "FPT", "CMG", "MWG", "PNVN", "HPG", "GAS", "PLX", "POW", "VNM", "SAB", "MSN",
    "VIC", "VPB", "CTG", "PNJ", "HDB", "STB", "TPB", "MSB", "SHB", "LPB", "EIB", "OCB",
    "IMP", "PLD", "PDR", "NVL", "BCM", "SBT", "DHG", "IMP", "KDC", "REE", "PC1", "HDG"
}

def get_yahoo_symbol(symbol: str) -> str:
    """Convert VN ticker to Yahoo Finance format (add .VN suffix)."""
    symbol = symbol.upper()
    if symbol in VN_SYMBOLS:
        return f"{symbol}.VN"
    return symbol


class StockScraperProvider(StockDataProvider):
    """Scrape stock data from public websites - no API required."""

    source_name = "web_scraper"

    # US Stock sources (free)
    SOURCES = {
        "us": {
            "finance": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            "site": "https://www.google.com/finance/quote/{symbol}:NYSE",
        },
        "vietnam": {
            "cafef": "https://s.cafef.vn/LichSuGiaoDich-{symbol}-1.chn",
            "vietstock": "https://finance.vietstock.vn/{symbol}/historical-data",
        },
    }

    def __init__(self, source: Literal["us", "vietnam"] = "us"):
        self.source = source
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def validate_symbol(self, symbol: str) -> bool:
        try:
            frame = self.get_latest_data(symbol)
            return not frame.empty
        except Exception:
            return False

    def get_historical_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        
        # Try Yahoo Finance API first (no auth needed for basic data)
        if self.source == "us":
            return self._fetch_yahoo_finance(symbol, start, end)
        
        return self._fetch_cafef(symbol, start, end)

    def _fetch_yahoo_finance(self, symbol: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        """Fetch from Yahoo Finance using their public endpoint."""
        end = end or datetime.now(timezone.utc)
        # Default to 10+ years of history for ML training
        start = start or end.replace(year=end.year - 12)
        
        # Convert to Unix timestamps
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        # Add .VN suffix for Vietnamese stocks
        yahoo_symbol = get_yahoo_symbol(symbol)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise DataSourceError(f"Yahoo Finance request failed: {exc}") from exc
        except ValueError as exc:
            raise DataSourceError(f"Invalid Yahoo Finance response: {exc}") from exc
        
        # Parse response
        if "chart" not in data or "result" not in data["chart"]:
            error = data.get("chart", {}).get("error", {})
            raise DataSourceError(f"Yahoo Finance error: {error.get('description', 'Unknown')}")
        
        result = data["chart"]["result"]
        if not result:
            raise DataSourceError(f"No data for {symbol}")
        
        result = result[0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adj_close = result["indicators"]["adjclose"][0]["adjclose"] if "adjclose" in result["indicators"] else quote["close"]
        
        records = []
        for i, ts in enumerate(timestamps):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            records.append({
                "timestamp": dt,
                "open": quote["open"][i],
                "high": quote["high"][i],
                "low": quote["low"][i],
                "close": quote["close"][i],
                "adj_close": adj_close[i],
                "volume": quote["volume"][i],
            })
        
        if not records:
            raise DataSourceError(f"No data returned for {symbol}")
        
        frame = pd.DataFrame(records)
        frame["symbol"] = symbol
        frame["source"] = self.source_name
        frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        
        # Filter by date range - use tz_convert for timezone handling
        if start:
            start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
            frame = frame[frame["timestamp"] >= start_ts]
        if end:
            end_ts = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
            frame = frame[frame["timestamp"] <= end_ts]
        
        logger.info("Scraped %s rows from Yahoo Finance for %s", len(frame), symbol)
        return frame

    def _fetch_cafef(self, symbol: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        """Fetch from CafeF (Vietnamese stock data)."""
        # Map US symbols to Vietnamese stock codes
        symbol_map = {
            "AAPL": "AAPL", "GOOGL": "GOOGL", "MSFT": "MSFT",
            "AMZN": "AMZN", "TSLA": "TSLA", "NVDA": "NVDA", "META": "META"
        }
        
        # For Vietnamese stocks (HOSE/HNX)
        vietnam_map = {
            "VNM": "VNM", "VIC": "VIC", "VPB": "VPB",
            "TCB": "TCB", "MBB": "MBB", "SSI": "SSI"
        }
        
        if symbol in vietnam_map:
            url = f"https://s.cafef.vn/LichSuGiaoDich-{vietnam_map[symbol]}-1.chn"
        else:
            # Fall back to Yahoo Finance for US stocks
            return self._fetch_yahoo_finance(symbol, start, end)
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Parse table from CafeF
            table = soup.find("table", {"id": "table1"})
            if not table:
                raise DataSourceError("Could not find data table on CafeF")
            
            records = []
            rows = table.find_all("tr")[1:]  # Skip header
            
            for row in rows[:365]:  # Max 1 year
                cells = row.find_all("td")
                if len(cells) >= 6:
                    try:
                        date_str = cells[0].get_text(strip=True)
                        dt = datetime.strptime(date_str, "%d/%m/%Y").replace(tzinfo=timezone.utc)
                        
                        records.append({
                            "timestamp": dt,
                            "open": float(cells[1].get_text(strip=True).replace(",", "")),
                            "high": float(cells[2].get_text(strip=True).replace(",", "")),
                            "low": float(cells[3].get_text(strip=True).replace(",", "")),
                            "close": float(cells[4].get_text(strip=True).replace(",", "")),
                            "adj_close": float(cells[4].get_text(strip=True).replace(",", "")),
                            "volume": float(cells[5].get_text(strip=True).replace(",", "").replace(".", "")),
                        })
                    except (ValueError, IndexError):
                        continue
            
            if not records:
                raise DataSourceError(f"No data scraped from CafeF for {symbol}")
            
            frame = pd.DataFrame(records)
            frame["symbol"] = symbol
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
            frame = frame[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
            
            logger.info("Scraped %s rows from CafeF for %s", len(frame), symbol)
            return frame
            
        except requests.exceptions.RequestException as exc:
            raise DataSourceError(f"CafeF request failed: {exc}") from exc

    def get_latest_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        end = datetime.now(timezone.utc).replace(tzinfo=None)  # Naive datetime
        start = end - pd.Timedelta(days=30)  # Last 30 days
        frame = self.get_historical_data(symbol=symbol, start=start, end=end, interval=interval)
        return frame.tail(5).reset_index(drop=True)
