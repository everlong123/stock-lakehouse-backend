"""Continuous stock data crawler service."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger
from app.data_sources import get_data_provider
from app.lakehouse.bronze import BronzeLayer

logger = get_logger(__name__)


class StockCrawler:
    """Background crawler that fetches and stores stock data continuously."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._provider = None
        self._symbols = [s.strip().upper() for s in settings.crawl_symbols.split(",") if s.strip()]
        self._interval = settings.crawl_interval_seconds
        self._last_fetch: dict[str, datetime] = {}
        self._callbacks: list[Callable[[str, pd.DataFrame], None]] = []

    def start(self):
        """Start the crawler in a background thread."""
        if self._running:
            logger.warning("Crawler already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="StockCrawler")
        self._thread.start()
        logger.info(f"StockCrawler started for symbols: {self._symbols}, interval: {self._interval}s")

    def stop(self):
        """Stop the crawler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("StockCrawler stopped")

    def add_callback(self, callback: Callable[[str, pd.DataFrame], None]):
        """Register a callback to be called when new data arrives."""
        self._callbacks.append(callback)

    def _run_loop(self):
        """Main crawler loop running in background thread."""
        while self._running:
            try:
                self.fetch_all()
            except Exception as e:
                logger.error(f"Crawler error: {e}")
            time.sleep(self._interval)

    def fetch_all(self):
        """Fetch data for all configured symbols."""
        if not self._provider:
            try:
                self._provider = get_data_provider()
            except Exception as e:
                logger.error(f"Failed to initialize data provider: {e}")
                return

        for symbol in self._symbols:
            try:
                self.fetch_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

    def fetch_symbol(self, symbol: str) -> pd.DataFrame:
        """Fetch and store data for a single symbol."""
        logger.info(f"Fetching {symbol}...")
        
        # ALWAYS fetch full history - ignore existing data to get fresh real data
        logger.info(f"Fetching FULL history for {symbol}...")
        end = datetime.now(timezone.utc)
        start = end - pd.Timedelta(days=settings.default_lookback_days)  # 730 days by default
        
        # Convert to naive datetime for provider compatibility
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)
        
        frame = self._provider.get_historical_data(symbol, start=start_naive, end=end_naive, interval="1d")
        self._last_fetch[symbol] = datetime.now(timezone.utc)
        
        if frame.empty:
            logger.warning(f"No data returned for {symbol}")
            return frame

        # Store in Bronze layer
        try:
            bronze = BronzeLayer()
            bronze.append(frame, lineage={"source": "crawler", "symbol": symbol})
            logger.info(f"Saved {len(frame)} rows to Bronze for {symbol}")
        except Exception as e:
            logger.error(f"Failed to write to lakehouse: {e}")

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(symbol, frame)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return frame

    @property
    def last_fetch_times(self) -> dict[str, datetime]:
        """Return dict of symbol -> last fetch time."""
        return self._last_fetch.copy()

    @property
    def is_running(self) -> bool:
        return self._running


# Global singleton instance
_crawler: StockCrawler | None = None


def get_crawler() -> StockCrawler:
    """Get or create the global crawler instance."""
    global _crawler
    if _crawler is None:
        _crawler = StockCrawler()
    return _crawler


def start_crawler():
    """Start the global crawler if enabled."""
    if settings.crawl_enabled:
        crawler = get_crawler()
        crawler.start()
        return crawler
    return None


def stop_crawler():
    """Stop the global crawler."""
    global _crawler
    if _crawler:
        _crawler.stop()
        _crawler = None
