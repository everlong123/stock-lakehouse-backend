"""Data ingestion service for all data sources (stocks, fundamental, news, macro)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.logging_config import get_logger
from app.data_sources import (
    FundamentalDataProvider,
    MacroDataProvider,
    NewsSentimentProvider,
    VNStockProvider,
    get_data_provider,
)
from app.lakehouse.bronze import BronzeLayer

logger = get_logger(__name__)


class DataIngestionService:
    """Centralized service for ingesting data from all sources."""
    
    def __init__(self):
        self.vn_provider = VNStockProvider()
        self.fundamental_provider = FundamentalDataProvider()
        self.news_provider = NewsSentimentProvider()
        self.macro_provider = MacroDataProvider()
        self.bronze = BronzeLayer()
    
    # =====================
    # VN Stock Data (OHLCV)
    # =====================
    
    def ingest_vn_stocks(self, symbols: list[str] | None = None) -> dict[str, int]:
        """Ingest VN stock data for given symbols or default indices."""
        if symbols is None:
            symbols = self.vn_provider.VN30_SYMBOLS
        
        results = {}
        for symbol in symbols:
            try:
                frame = self.vn_provider.get_historical_data(symbol)
                if not frame.empty:
                    self.bronze.append(frame, lineage={
                        "source": "vn_stock",
                        "symbol": symbol,
                        "type": "ohlcv"
                    })
                    results[symbol] = len(frame)
                    logger.info(f"Ingested {len(frame)} rows for {symbol}")
                else:
                    results[symbol] = 0
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
                results[symbol] = -1
        
        return results
    
    def ingest_index_components(
        self, 
        index: str = "VN30"
    ) -> dict[str, int]:
        """Ingest all stocks for a VN index."""
        symbols = self.vn_provider.get_index_components(index)
        return self.ingest_vn_stocks(symbols)
    
    # =====================
    # Fundamental Data
    # =====================
    
    def ingest_fundamental(self, symbols: list[str]) -> dict[str, Any]:
        """Ingest fundamental data for symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fundamental_provider.get_financial_summary(symbol)
                
                if data:
                    # Store fundamental data
                    frame = pd.DataFrame([data])
                    self.bronze.append(frame, lineage={
                        "source": "fundamental",
                        "symbol": symbol,
                        "type": "financial_summary"
                    })
                    results[symbol] = "success"
                    logger.info(f"Ingested fundamental data for {symbol}")
                else:
                    results[symbol] = "no_data"
                    
            except Exception as e:
                logger.error(f"Failed fundamental for {symbol}: {e}")
                results[symbol] = "error"
        
        return results
    
    # =====================
    # News + Sentiment
    # =====================
    
    def ingest_news(self, symbols: list[str] | None = None, limit: int = 20) -> pd.DataFrame:
        """Ingest news and sentiment data."""
        all_news = []
        
        if symbols:
            for symbol in symbols:
                try:
                    news = self.news_provider.get_news_for_symbol(symbol, limit)
                    if not news.empty:
                        all_news.append(news)
                except Exception as e:
                    logger.error(f"News ingest failed for {symbol}: {e}")
        else:
            # Market news
            try:
                news = self.news_provider.get_market_news(limit)
                if not news.empty:
                    all_news.append(news)
            except Exception as e:
                logger.error(f"Market news ingest failed: {e}")
        
        if all_news:
            combined = pd.concat(all_news, ignore_index=True)
            self.bronze.append(combined, lineage={
                "source": "news_sentiment",
                "type": "news_articles"
            })
            logger.info(f"Ingested {len(combined)} news articles")
            return combined
        
        return pd.DataFrame()
    
    def ingest_sentiment_summary(self, symbols: list[str]) -> pd.DataFrame:
        """Ingest aggregated sentiment summaries."""
        summaries = []
        
        for symbol in symbols:
            try:
                summary = self.news_provider.get_sentiment_summary(symbol)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Sentiment failed for {symbol}: {e}")
        
        if summaries:
            frame = pd.DataFrame(summaries)
            self.bronze.append(frame, lineage={
                "source": "news_sentiment",
                "type": "sentiment_summary"
            })
            logger.info(f"Ingested sentiment for {len(summaries)} symbols")
            return frame
        
        return pd.DataFrame()
    
    # =====================
    # Macro Data
    # =====================
    
    def ingest_exchange_rates(self) -> pd.DataFrame:
        """Ingest USD/VND and other exchange rates."""
        try:
            frame = self.macro_provider.get_exchange_rate()
            if not frame.empty:
                self.bronze.append(frame, lineage={
                    "source": "macro",
                    "type": "exchange_rates"
                })
                logger.info(f"Ingested {len(frame)} exchange rate records")
            return frame
        except Exception as e:
            logger.error(f"Exchange rate ingest failed: {e}")
            return pd.DataFrame()
    
    def ingest_gold_prices(self) -> pd.DataFrame:
        """Ingest gold prices."""
        try:
            frame = self.macro_provider.get_gold_price()
            if not frame.empty:
                self.bronze.append(frame, lineage={
                    "source": "macro",
                    "type": "gold_prices"
                })
                logger.info(f"Ingested {len(frame)} gold price records")
            return frame
        except Exception as e:
            logger.error(f"Gold price ingest failed: {e}")
            return pd.DataFrame()
    
    def ingest_oil_prices(self) -> pd.DataFrame:
        """Ingest oil prices (WTI, Brent)."""
        try:
            frame = self.macro_provider.get_oil_price()
            if not frame.empty:
                self.bronze.append(frame, lineage={
                    "source": "macro",
                    "type": "oil_prices"
                })
                logger.info(f"Ingested {len(frame)} oil price records")
            return frame
        except Exception as e:
            logger.error(f"Oil price ingest failed: {e}")
            return pd.DataFrame()
    
    def ingest_interest_rates(self) -> pd.DataFrame:
        """Ingest interest rates."""
        try:
            frame = self.macro_provider.get_interest_rates()
            if not frame.empty:
                self.bronze.append(frame, lineage={
                    "source": "macro",
                    "type": "interest_rates"
                })
                logger.info(f"Ingested {len(frame)} interest rate records")
            return frame
        except Exception as e:
            logger.error(f"Interest rate ingest failed: {e}")
            return pd.DataFrame()
    
    def ingest_cpi(self) -> pd.DataFrame:
        """Ingest CPI data."""
        try:
            frame = self.macro_provider.get_cpi()
            if not frame.empty:
                self.bronze.append(frame, lineage={
                    "source": "macro",
                    "type": "cpi"
                })
                logger.info(f"Ingested {len(frame)} CPI records")
            return frame
        except Exception as e:
            logger.error(f"CPI ingest failed: {e}")
            return pd.DataFrame()
    
    def ingest_all_macro(self) -> dict[str, Any]:
        """Ingest all macro indicators."""
        results = {
            "exchange_rates": 0,
            "gold": 0,
            "oil": 0,
            "interest_rates": 0,
            "cpi": 0,
        }
        
        fx = self.ingest_exchange_rates()
        results["exchange_rates"] = len(fx)
        
        gold = self.ingest_gold_prices()
        results["gold"] = len(gold)
        
        oil = self.ingest_oil_prices()
        results["oil"] = len(oil)
        
        rates = self.ingest_interest_rates()
        results["interest_rates"] = len(rates)
        
        cpi = self.ingest_cpi()
        results["cpi"] = len(cpi)
        
        return results
    
    # =====================
    # Full Pipeline
    # =====================
    
    def ingest_all(self) -> dict[str, Any]:
        """Run full data ingestion for all sources."""
        logger.info("Starting full data ingestion...")
        
        summary = {
            "vn_stocks": {},
            "fundamental": {},
            "news": 0,
            "macro": {},
            "errors": [],
        }
        
        # 1. VN Stocks
        try:
            summary["vn_stocks"] = self.ingest_index_components("VN30")
        except Exception as e:
            summary["errors"].append(f"VN Stocks: {e}")
        
        # 2. Fundamental
        try:
            symbols = list(self.vn_provider.VN30_SYMBOLS[:10])  # First 10 for demo
            summary["fundamental"] = self.ingest_fundamental(symbols)
        except Exception as e:
            summary["errors"].append(f"Fundamental: {e}")
        
        # 3. News
        try:
            news = self.ingest_news(limit=30)
            summary["news"] = len(news)
        except Exception as e:
            summary["errors"].append(f"News: {e}")
        
        # 4. Macro
        try:
            summary["macro"] = self.ingest_all_macro()
        except Exception as e:
            summary["errors"].append(f"Macro: {e}")
        
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Full ingestion completed. Errors: {len(summary['errors'])}")
        
        return summary


# Singleton instance
_ingestion_service: DataIngestionService | None = None


def get_ingestion_service() -> DataIngestionService:
    """Get or create the ingestion service singleton."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DataIngestionService()
    return _ingestion_service
