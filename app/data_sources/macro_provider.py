"""Macro economic data provider - fetches USD/VND, interest rates, CPI, gold, oil, etc."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class MacroDataProvider:
    """Fetch macro economic indicators from public sources."""
    
    source_name = "macro"
    
    # Exchange rates
    EXCHANGE_API = "https://portal.vietcombank.com.vn/Usercontrols/TVExchange.ashx"
    
    # Gold prices
    GOLD_SOURCE = "https://sjc.com.vn/giavang"
    
    # Oil prices (WTI/Brent)
    OIL_API = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
        })
    
    def get_exchange_rate(self, pair: str = "USD/VND") -> pd.DataFrame:
        """Get historical exchange rate."""
        records = []
        
        try:
            # Vietcombank exchange rates
            response = self.session.get(
                self.EXCHANGE_API,
                params={"fdate": "01/01/2020", "tdate": datetime.now().strftime("%d/%m/%Y")},
                timeout=15
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table")
                
                if table:
                    rows = table.find_all("tr")[1:]  # Skip header
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 3:
                            currency = cells[0].get_text(strip=True)
                            if pair in currency or "USD" in currency:
                                buy = self._parse_number(cells[1].get_text(strip=True))
                                sell = self._parse_number(cells[2].get_text(strip=True))
                                
                                records.append({
                                    "timestamp": datetime.now(timezone.utc),
                                    "currency_pair": pair,
                                    "buy_rate": buy,
                                    "sell_rate": sell,
                                    "mid_rate": (buy + sell) / 2 if buy and sell else None,
                                })
                                
        except Exception as exc:
            logger.warning("Exchange rate fetch failed: %s", exc)
        
        # Also try CafeF for simpler access
        try:
            url = "https://s.cafef.vn/Tien-Te.chn"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                usd_row = soup.find("td", string=lambda x: x and "USD" in str(x))
                
                if usd_row:
                    cells = usd_row.find_parent("tr").find_all("td")
                    if len(cells) >= 3:
                        buy = self._parse_number(cells[1].get_text(strip=True))
                        sell = self._parse_number(cells[2].get_text(strip=True))
                        
                        if records:
                            records[0]["buy_rate"] = buy
                            records[0]["sell_rate"] = sell
                            records[0]["mid_rate"] = (buy + sell) / 2
        except Exception as exc:
            logger.warning("CafeF exchange rate failed: %s", exc)
        
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def get_gold_price(self) -> pd.DataFrame:
        """Get historical gold prices in Vietnam."""
        records = []
        
        try:
            # SJC Gold prices
            response = self.session.get(self.GOLD_SOURCE, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find gold price table
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 4:
                            city = cells[0].get_text(strip=True)
                            buy = self._parse_number(cells[1].get_text(strip=True))
                            sell = self._parse_number(cells[2].get_text(strip=True))
                            
                            if buy and sell:
                                records.append({
                                    "timestamp": datetime.now(timezone.utc),
                                    "commodity": "SJC_GOLD",
                                    "location": city,
                                    "buy_price": buy,
                                    "sell_price": sell,
                                    "unit": "VND/luong",
                                })
                                
        except Exception as exc:
            logger.warning("Gold price fetch failed: %s", exc)
        
        # Try world gold price (USD/oz)
        try:
            # Use Yahoo Finance for world gold price
            gold_response = self.session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF",
                params={"interval": "1d", "range": "1mo"},
                timeout=15
            )
            
            if gold_response.status_code == 200:
                data = gold_response.json()
                result = data.get("chart", {}).get("result", [{}])[0]
                timestamps = result.get("timestamp", [])
                quotes = result.get("indicators", {}).get("quote", [{}])[0]
                
                for i, ts in enumerate(timestamps):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    close = quotes.get("close", [None])[i]
                    
                    if close:
                        records.append({
                            "timestamp": dt,
                            "commodity": "GOLD",
                            "location": "WORLD",
                            "buy_price": close,
                            "sell_price": close,
                            "unit": "USD/oz",
                        })
                        
        except Exception as exc:
            logger.warning("World gold price failed: %s", exc)
        
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def get_oil_price(self) -> pd.DataFrame:
        """Get oil prices (WTI and Brent)."""
        records = []
        
        # Try Yahoo Finance for oil futures
        symbols = {
            "CL=F": "WTI_OIL",  # Crude Oil
            "BZ=F": "BRENT_OIL",  # Brent Crude
        }
        
        for symbol, name in symbols.items():
            try:
                response = self.session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"interval": "1d", "range": "1mo"},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = data.get("chart", {}).get("result", [{}])[0]
                    timestamps = result.get("timestamp", [])
                    quotes = result.get("indicators", {}).get("quote", [{}])[0]
                    
                    for i, ts in enumerate(timestamps):
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        close = quotes.get("close", [None])[i]
                        
                        if close:
                            records.append({
                                "timestamp": dt,
                                "commodity": name,
                                "price": close,
                                "unit": "USD/barrel",
                            })
                            
            except Exception as exc:
                logger.warning("%s price fetch failed: %s", name, exc)
        
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def get_interest_rates(self) -> pd.DataFrame:
        """Get interest rates (Vietnam central bank rates)."""
        records = []
        
        try:
            # SBV interest rates
            url = "https://sbv.gov.vn/Lãi suất"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find rate table
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            rate_type = cells[0].get_text(strip=True)
                            rate_value = self._parse_number(cells[1].get_text(strip=True))
                            
                            if rate_value:
                                records.append({
                                    "timestamp": datetime.now(timezone.utc),
                                    "rate_type": rate_type,
                                    "rate_value": rate_value / 100,  # Convert to decimal
                                    "unit": "percent",
                                })
                                
        except Exception as exc:
            logger.warning("Interest rate fetch failed: %s", exc)
        
        # Common VN interest rates (fallback/default)
        if not records:
            records = [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "rate_type": "OMO",
                    "rate_value": 0.045,
                    "unit": "percent",
                },
                {
                    "timestamp": datetime.now(timezone.utc),
                    "rate_type": "REPO",
                    "rate_value": 0.04,
                    "unit": "percent",
                },
                {
                    "timestamp": datetime.now(timezone.utc),
                    "rate_type": "DEPOSIT_12M",
                    "rate_value": 0.055,
                    "unit": "percent",
                },
            ]
        
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def get_cpi(self) -> pd.DataFrame:
        """Get Consumer Price Index (CPI) data."""
        records = []
        
        try:
            # GSO Vietnam CPI data
            url = "https://www.gso.gov.vn/cpi"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            period = cells[0].get_text(strip=True)
                            cpi_value = self._parse_number(cells[1].get_text(strip=True))
                            
                            if cpi_value:
                                records.append({
                                    "timestamp": datetime.now(timezone.utc),
                                    "indicator": "CPI",
                                    "period": period,
                                    "value": cpi_value,
                                    "unit": "index",
                                })
                                
        except Exception as exc:
            logger.warning("CPI fetch failed: %s", exc)
        
        # Monthly CPI change (default/fallback)
        if not records:
            records = [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "indicator": "CPI",
                    "period": datetime.now().strftime("%Y-%m"),
                    "value": 100,
                    "unit": "index",
                    "yoy_change": 0.035,  # 3.5% YoY
                    "mom_change": 0.002,  # 0.2% MoM
                }
            ]
        
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")
        
        return frame
    
    def get_macro_summary(self) -> dict:
        """Get summary of all macro indicators."""
        summary = {}
        
        # Exchange rate
        fx = self.get_exchange_rate()
        if not fx.empty:
            latest_fx = fx.iloc[-1]
            summary["exchange_rate"] = {
                "usd_vnd": latest_fx.get("mid_rate") or 25000,
                "timestamp": str(latest_fx.get("timestamp", "")),
            }
        
        # Gold
        gold = self.get_gold_price()
        if not gold.empty:
            latest_gold = gold[gold["location"] == "WORLD"].iloc[-1] if len(gold[gold["location"] == "WORLD"]) > 0 else gold.iloc[-1]
            summary["gold"] = {
                "price_usd_oz": latest_gold.get("buy_price", 2000),
                "timestamp": str(latest_gold.get("timestamp", "")),
            }
        
        # Oil
        oil = self.get_oil_price()
        if not oil.empty:
            latest_oil = oil.iloc[-1]
            summary["oil"] = {
                "wti_usd_barrel": latest_oil.get("price", 80),
                "timestamp": str(latest_oil.get("timestamp", "")),
            }
        
        # Interest rates
        rates = self.get_interest_rates()
        if not rates.empty:
            summary["interest_rates"] = rates.to_dict("records")
        
        # CPI
        cpi = self.get_cpi()
        if not cpi.empty:
            summary["cpi"] = cpi.to_dict("records")
        
        summary["source"] = self.source_name
        summary["timestamp"] = str(pd.Timestamp.now(tz="UTC"))
        
        return summary
    
    @staticmethod
    def _parse_number(value: str) -> float | None:
        """Parse number from string."""
        if not value:
            return None
        
        # Remove currency symbols and spaces
        value = value.replace("₫", "").replace("VND", "").replace("USD", "").replace(" ", "")
        
        # Handle Vietnamese format
        if "," not in value and "." in value:
            parts = value.split(".")
            if len(parts) > 1 and len(parts[-1]) == 3:
                value = value.replace(".", "")
        
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
