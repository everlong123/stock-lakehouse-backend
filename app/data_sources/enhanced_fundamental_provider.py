"""Enhanced Fundamental data provider - comprehensive financial metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FinancialMetrics:
    """Comprehensive financial metrics for a company."""
    symbol: str
    timestamp: datetime

    # Valuation
    market_cap: float | None
    pe_ratio: float | None
    pb_ratio: float | None
    ps_ratio: float | None
    peg_ratio: float | None
    ev_ebitda: float | None
    dividend_yield: float | None

    # Profitability
    roe: float | None  # Return on Equity
    roa: float | None  # Return on Assets
    roic: float | None  # Return on Invested Capital
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None

    # Growth
    revenue_growth_yoy: float | None
    profit_growth_yoy: float | None
    eps_growth_yoy: float | None

    # Financial Health
    debt_equity: float | None
    current_ratio: float | None
    quick_ratio: float | None

    # Per Share
    eps: float | None  # Earnings per share
    bvps: float | None  # Book value per share
    dps: float | None  # Dividend per share


class EnhancedFundamentalProvider:
    """Enhanced fundamental data provider with comprehensive financial metrics."""

    source_name = "fundamental_enhanced"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/html",
        })

    def get_comprehensive_metrics(self, symbol: str) -> FinancialMetrics:
        """
        Get comprehensive financial metrics for a symbol.

        Returns FinancialMetrics dataclass with all key metrics.
        """
        symbol = symbol.upper()

        # Try multiple sources
        metrics = self._fetch_vietstock_metrics(symbol)

        if metrics is None:
            metrics = self._fetch_yahoo_finance_metrics(symbol)

        if metrics is None:
            metrics = FinancialMetrics(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                market_cap=None, pe_ratio=None, pb_ratio=None, ps_ratio=None,
                peg_ratio=None, ev_ebitda=None, dividend_yield=None,
                roe=None, roa=None, roic=None, gross_margin=None,
                operating_margin=None, net_margin=None,
                revenue_growth_yoy=None, profit_growth_yoy=None, eps_growth_yoy=None,
                debt_equity=None, current_ratio=None, quick_ratio=None,
                eps=None, bvps=None, dps=None,
            )

        return metrics

    def _fetch_vietstock_metrics(self, symbol: str) -> FinancialMetrics | None:
        """Fetch from VietStock."""
        try:
            # Financial ratios page
            url = f"https://finance.vietstock.vn/{symbol}/financial-ratio"

            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Parse valuation ratios
            metrics = FinancialMetrics(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                market_cap=self._find_metric(soup, "thanh khoản", "vốn hóa"),
                pe_ratio=self._find_metric(soup, "hệ số", "P/E"),
                pb_ratio=self._find_metric(soup, "hệ số", "P/B"),
                ps_ratio=self._find_metric(soup, "hệ số", "P/S"),
                peg_ratio=self._find_metric(soup, "hệ số", "PEG"),
                ev_ebitda=self._find_metric(soup, "hệ số", "EV/EBITDA"),
                dividend_yield=self._find_metric(soup, "cổ tức", "cổ tức"),
                roe=self._find_metric(soup, "hiệu quả", "ROE"),
                roa=self._find_metric(soup, "hiệu quả", "ROA"),
                roic=None,
                gross_margin=self._find_metric(soup, "biên", "lợi nhuận gộp"),
                operating_margin=self._find_metric(soup, "biên", "hoạt động"),
                net_margin=self._find_metric(soup, "biên", "lợi nhuận ròng"),
                revenue_growth_yoy=self._find_metric(soup, "tăng trưởng", "doanh thu"),
                profit_growth_yoy=self._find_metric(soup, "tăng trưởng", "lợi nhuận"),
                eps_growth_yoy=None,
                debt_equity=self._find_metric(soup, "cơ cấu", "nợ/vốn"),
                current_ratio=self._find_metric(soup, "thanh khoản", "hiện tại"),
                quick_ratio=self._find_metric(soup, "thanh khoản", "nhanh"),
                eps=self._find_metric(soup, "cổ phiếu", "EPS"),
                bvps=self._find_metric(soup, "cổ phiếu", "BVPS"),
                dps=None,
            )

            return metrics

        except Exception as exc:
            logger.debug(f"VietStock fetch failed for {symbol}: {exc}")
            return None

    def _fetch_yahoo_finance_metrics(self, symbol: str) -> FinancialMetrics | None:
        """Fetch from Yahoo Finance summary."""
        try:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            params = {"modules": "summaryDetail,defaultKeyStatistics,financialData,assetProfile"}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return None

            data = response.json()
            result = data.get("quoteSummary", {}).get("result", [{}])

            if not result:
                return None

            summary = result[0].get("summaryDetail", {})
            key_stats = result[0].get("defaultKeyStatistics", {})
            financial = result[0].get("financialData", {})

            return FinancialMetrics(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                market_cap=summary.get("marketCap", {}).get("raw"),
                pe_ratio=summary.get("trailingPE", {}).get("raw"),
                pb_ratio=key_stats.get("priceBook", {}).get("raw"),
                ps_ratio=summary.get("priceToSalesTrailing12Months", {}).get("raw"),
                peg_ratio=key_stats.get("pegRatio", {}).get("raw"),
                ev_ebitda=financial.get("enterpriseToEbitda", {}).get("raw"),
                dividend_yield=summary.get("dividendYield", {}).get("raw"),
                roe=financial.get("returnOnEquity", {}).get("raw"),
                roa=financial.get("returnOnAssets", {}).get("raw"),
                roic=None,
                gross_margin=financial.get("grossMargins", {}).get("raw"),
                operating_margin=financial.get("operatingMargins", {}).get("raw"),
                net_margin=financial.get("netMargins", {}).get("raw"),
                revenue_growth_yoy=financial.get("revenueGrowth", {}).get("raw"),
                profit_growth_yoy=financial.get("profitGrowth", {}).get("raw"),
                eps_growth_yoy=key_stats.get("earningsQuarterlyGrowth", {}).get("raw"),
                debt_equity=financial.get("debtToEquity", {}).get("raw"),
                current_ratio=financial.get("currentRatio", {}).get("raw"),
                quick_ratio=financial.get("quickRatio", {}).get("raw"),
                eps=key_stats.get("trailingEps", {}).get("raw"),
                bvps=key_stats.get("bookValue", {}).get("raw"),
                dps=key_stats.get("dividendRate", {}).get("raw"),
            )

        except Exception as exc:
            logger.debug(f"Yahoo Finance fetch failed for {symbol}: {exc}")
            return None

    def get_income_statement(self, symbol: str, years: int = 4) -> pd.DataFrame:
        """Get historical income statement data."""
        records = []

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1mo", "range": f"{years}y"}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [{}])

                if result:
                    financials = result[0].get("incomeStatementHistory", {}).get("incomeStatementHistory", [])

                    for period in financials[:years]:
                        for date_str, values in period.items():
                            records.append({
                                "symbol": symbol.upper(),
                                "period": date_str,
                                "timestamp": datetime.fromisoformat(date_str.replace(" ", "T")).replace(tzinfo=timezone.utc),
                                "total_revenue": values.get("totalRevenue", {}).get("raw"),
                                "gross_profit": values.get("grossProfit", {}).get("raw"),
                                "operating_income": values.get("operatingIncome", {}).get("raw"),
                                "net_income": values.get("netIncome", {}).get("raw"),
                                "ebit": values.get("ebit", {}).get("raw"),
                                "eps": values.get("netIncomeCommonStockholders", {}).get("raw"),
                            })

        except Exception as exc:
            logger.warning(f"Income statement fetch failed for {symbol}: {exc}")

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")

        return frame

    def get_balance_sheet(self, symbol: str, years: int = 4) -> pd.DataFrame:
        """Get historical balance sheet data."""
        records = []

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1mo", "range": f"{years}y"}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [{}])

                if result:
                    balance = result[0].get("balanceSheetHistory", {}).get("balanceSheetStatements", [])

                    for period in balance[:years]:
                        for date_str, values in period.items():
                            records.append({
                                "symbol": symbol.upper(),
                                "period": date_str,
                                "timestamp": datetime.fromisoformat(date_str.replace(" ", "T")).replace(tzinfo=timezone.utc),
                                "total_assets": values.get("totalAssets", {}).get("raw"),
                                "total_liabilities": values.get("totalLiab", {}).get("raw"),
                                "total_equity": values.get("totalStockholderEquity", {}).get("raw"),
                                "cash": values.get("cash", {}).get("raw"),
                                "short_term_debt": values.get("shortLongTermDebt", {}).get("raw"),
                                "long_term_debt": values.get("longTermDebt", {}).get("raw"),
                            })

        except Exception as exc:
            logger.warning(f"Balance sheet fetch failed for {symbol}: {exc}")

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")

        return frame

    def get_cash_flow(self, symbol: str, years: int = 4) -> pd.DataFrame:
        """Get historical cash flow data."""
        records = []

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1mo", "range": f"{years}y"}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [{}])

                if result:
                    cashflow = result[0].get("cashflowStatementHistory", {}).get("cashflowStatements", [])

                    for period in cashflow[:years]:
                        for date_str, values in period.items():
                            records.append({
                                "symbol": symbol.upper(),
                                "period": date_str,
                                "timestamp": datetime.fromisoformat(date_str.replace(" ", "T")).replace(tzinfo=timezone.utc),
                                "operating_cashflow": values.get("totalCashFromOperatingActivities", {}).get("raw"),
                                "capital_expenditure": values.get("capitalExpenditures", {}).get("raw"),
                                "free_cashflow": values.get("freeCashFlow", {}).get("raw"),
                                "dividends_paid": values.get("dividendsPaid", {}).get("raw"),
                                "share_repurchase": values.get("commonStockRepurchased", {}).get("raw"),
                            })

        except Exception as exc:
            logger.warning(f"Cash flow fetch failed for {symbol}: {exc}")

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["source"] = self.source_name
            frame["ingestion_time"] = pd.Timestamp.now(tz="UTC")

        return frame

    def metrics_to_dict(self, metrics: FinancialMetrics) -> dict[str, Any]:
        """Convert FinancialMetrics to dictionary."""
        result = {
            "symbol": metrics.symbol,
            "timestamp": metrics.timestamp.isoformat(),
        }

        # Valuation
        result.update({
            "market_cap": metrics.market_cap,
            "pe_ratio": metrics.pe_ratio,
            "pb_ratio": metrics.pb_ratio,
            "ps_ratio": metrics.ps_ratio,
            "peg_ratio": metrics.peg_ratio,
            "ev_ebitda": metrics.ev_ebitda,
            "dividend_yield": metrics.dividend_yield,
        })

        # Profitability
        result.update({
            "roe": metrics.roe,
            "roa": metrics.roa,
            "roic": metrics.roic,
            "gross_margin": metrics.gross_margin,
            "operating_margin": metrics.operating_margin,
            "net_margin": metrics.net_margin,
        })

        # Growth
        result.update({
            "revenue_growth_yoy": metrics.revenue_growth_yoy,
            "profit_growth_yoy": metrics.profit_growth_yoy,
            "eps_growth_yoy": metrics.eps_growth_yoy,
        })

        # Financial Health
        result.update({
            "debt_equity": metrics.debt_equity,
            "current_ratio": metrics.current_ratio,
            "quick_ratio": metrics.quick_ratio,
        })

        # Per Share
        result.update({
            "eps": metrics.eps,
            "bvps": metrics.bvps,
            "dps": metrics.dps,
        })

        return result

    def _find_metric(self, soup: BeautifulSoup, section: str, metric: str) -> float | None:
        """Find a metric value from parsed HTML."""
        # This is a simplified implementation
        # In production, you'd use more robust parsing
        try:
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        if metric.lower() in label:
                            return self._parse_number(cells[1].get_text(strip=True))
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_number(value: str) -> float | None:
        """Parse number from string."""
        if not value:
            return None

        # Remove percentage sign and convert
        value = value.replace("%", "").replace(",", "").strip()

        try:
            return float(value)
        except ValueError:
            return None
