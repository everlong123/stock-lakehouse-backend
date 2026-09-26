"""Factory for the configured market data provider."""

from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import get_logger
from app.data_sources.base import StockDataProvider
from app.data_sources.sample_provider import SampleDataProvider
from app.data_sources.yfinance_provider import YFinanceProvider
from app.data_sources.alpha_vantage_provider import AlphaVantageProvider
from app.data_sources.web_scraper_provider import StockScraperProvider

logger = get_logger(__name__)


def get_data_provider(name: str | None = None) -> StockDataProvider:
    """Return the provider selected by ENV or explicit name."""
    selected = (name or settings.data_source).lower()
    if selected == "yfinance":
        logger.info("Using YFinanceProvider (near-real-time polling, not tick-level).")
        return YFinanceProvider()
    if selected == "alpha_vantage":
        logger.info("Using AlphaVantageProvider.")
        return AlphaVantageProvider()
    if selected == "web_scraper":
        logger.info("Using StockScraperProvider (no API key required).")
        return StockScraperProvider()
    logger.info("Using SampleDataProvider (offline demo dataset).")
    return SampleDataProvider()
