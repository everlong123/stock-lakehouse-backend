"""Fundamental data provider for financial metrics (Revenue, EPS, ROE, P/E, P/B, etc.)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class FundamentalDataProvider:
    """Fetch fundamental financial data from public sources."""
    
    source_name = "fundamental"
    
    # VietStock API for financial data
    VIETSTOCK_API = "https://finance.vietstock.vn/{symbol}/financial-report"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
        })
    
    def get_financial_summary(self, symbol: str) -> dict[str, Any]:
        """Get key financial metrics for a symbol."""
        symbol = symbol.upper()
        
        # Try VietStock for VN stocks
        data = self._fetch_vietstock_financial(symbol)
        
        if data:
            return {
                "symbol": symbol,
                "revenue": data.get("revenue"),
                "net_profit": data.get("net_profit"),
                "eps": data.get("eps"),
                "roe": data.get("roe"),
                "roa": data.get("roa"),
                "pe_ratio": data.get("pe"),
                "pb_ratio": data.get("pb"),
                "market_cap": data.get("market_cap"),
                "source": self.source_name,
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }
        
        return {}
    
    def _fetch_vietstock_financial(self, symbol: str) -> dict | None:
        """Fetch from VietStock."""
        try:
            url = f"https://finance.vietstock.vn/{symbol}/company-financial"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return None
            
            # Parse HTML for financial data
            soup = BeautifulSoup(response.text, "html.parser")
            
            data = {}
            
            # Try to find common financial metrics in the page
            # This is a simplified version - actual implementation would need more robust parsing
            tables = soup.find_all("table", {"class": "financial-table"})
            
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        if "eps" in label:
                            data["eps"] = self._parse_number(value)
                        elif "roe" in label:
                            data["roe"] = self._parse_number(value)
                        elif "roa" in label:
                            data["roa"] = self._parse_number(value)
                        elif "p/e" in label or "pe" in label:
                            data["pe"] = self._parse_number(value)
                        elif "p/b" in label or "pb" in label:
                            data["pb"] = self._parse_number(value)
                        elif "doanh thu" in label or "revenue" in label:
                            data["revenue"] = self._parse_number(value)
                        elif "lợi nhuận" in label or "profit" in label:
                            data["net_profit"] = self._parse_number(value)
            
            return data if data else None
            
        except Exception as exc:
            logger.warning("Failed to fetch VietStock financial for %s: %s", symbol, exc)
            return None
    
    def get_income_statement(self, symbol: str, year: int | None = None) -> pd.DataFrame:
        """Get income statement data."""
        return pd.DataFrame()
    
    def get_balance_sheet(self, symbol: str, year: int | None = None) -> pd.DataFrame:
        """Get balance sheet data."""
        return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str, year: int | None = None) -> pd.DataFrame:
        """Get cash flow statement data."""
        return pd.DataFrame()
    
    def get_ratios(self, symbol: str) -> pd.DataFrame:
        """Get financial ratios over time."""
        return pd.DataFrame()
    
    @staticmethod
    def _parse_number(value: str) -> float | None:
        """Parse number from string (handles Vietnamese format)."""
        if not value:
            return None
        
        # Remove currency symbols, spaces, and common separators
        value = value.replace("₫", "").replace("$", "").replace(" ", "")
        
        # Handle Vietnamese number format (1.234.567 -> 1234567)
        if "," not in value and "." in value:
            # Check if it's Vietnamese format (dots as thousand separators)
            parts = value.split(".")
            if len(parts) > 1 and len(parts[-1]) == 3:
                value = value.replace(".", "")
        
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
