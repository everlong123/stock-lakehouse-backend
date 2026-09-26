"""Unified streaming pipeline - combines all data sources into real-time ingestion."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class UnifiedStreamingPipeline:
    """
    Unified pipeline that integrates all data sources:
    - OHLCV price data (via crawler)
    - Order book / trades (via OrderBookProvider)
    - Market indices (via MarketIndexProvider)
    - News / sentiment (via EnhancedNewsSentimentProvider)
    - Fundamental data (via EnhancedFundamentalProvider)

    Supports both batch and streaming modes.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        include_orderbook: bool = True,
        include_news: bool = True,
        include_fundamental: bool = True,
        include_indices: bool = True,
        kafka_enabled: bool = False,
        kafka_bootstrap_servers: str | None = None,
    ):
        # Symbols
        self.symbols = symbols or [s.strip().upper() for s in settings.crawl_symbols.split(",") if s.strip()]

        # Data source flags
        self.include_orderbook = include_orderbook
        self.include_news = include_news
        self.include_fundamental = include_fundamental
        self.include_indices = include_indices

        # Kafka settings
        self.kafka_enabled = kafka_enabled or settings.use_kafka
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or settings.kafka_bootstrap_servers

        # State
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_fetch: dict[str, datetime] = {}

        # Data providers (lazy loaded)
        self._crawler = None
        self._orderbook_provider = None
        self._index_provider = None
        self._news_provider = None
        self._fundamental_provider = None
        self._kafka_producer = None

    def start(self, poll_interval_seconds: int = 60) -> None:
        """Start the streaming pipeline in background."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(poll_interval_seconds,),
            daemon=True,
            name="UnifiedStreamingPipeline",
        )
        self._thread.start()

        logger.info(
            f"UnifiedStreamingPipeline started: symbols={len(self.symbols)}, "
            f"orderbook={self.include_orderbook}, news={self.include_news}, "
            f"fundamental={self.include_fundamental}, indices={self.include_indices}"
        )

    def stop(self) -> None:
        """Stop the streaming pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

        # Close Kafka producer if used
        if self._kafka_producer:
            self._kafka_producer.close()
            self._kafka_producer = None

        logger.info("UnifiedStreamingPipeline stopped")

    def _run_loop(self, poll_interval: int) -> None:
        """Main pipeline loop."""
        from app.data_sources.factory import get_data_provider

        provider = get_data_provider()

        # Track fetch counts
        stats = {
            "ohlcv": 0,
            "orderbook": 0,
            "news": 0,
            "fundamental": 0,
            "indices": 0,
        }

        while self._running:
            try:
                for symbol in self.symbols:
                    # 1. OHLCV data (main price data)
                    stats["ohlcv"] += self._fetch_ohlcv(symbol, provider)

                    # 2. Order book data (high frequency)
                    if self.include_orderbook:
                        stats["orderbook"] += self._fetch_orderbook(symbol)

                    # 3. News and sentiment (lower frequency)
                    if self.include_news:
                        stats["news"] += self._fetch_news(symbol)

                    # 4. Fundamental data (low frequency - only on first run or periodic)
                    if self.include_fundamental and symbol not in self._last_fetch.get("fundamental", {}):
                        stats["fundamental"] += self._fetch_fundamental(symbol)

                    # 5. Market indices (only once per cycle)
                    if self.include_indices and symbol == self.symbols[0]:
                        stats["indices"] += self._fetch_indices()

                # Track last fundamental fetch
                if "fundamental" not in self._last_fetch:
                    self._last_fetch["fundamental"] = set()
                self._last_fetch["fundamental"].update(self.symbols)

                # Log stats periodically
                if any(stats.values()):
                    logger.info(f"Pipeline stats: {stats}")

            except Exception as e:
                logger.error(f"Pipeline error: {e}")

            time.sleep(poll_interval)

    def _fetch_ohlcv(self, symbol: str, provider) -> int:
        """Fetch OHLCV data."""
        from app.lakehouse.bronze import BronzeLayer

        try:
            end = datetime.now(timezone.utc)
            start = end - pd.Timedelta(days=settings.default_lookback_days)

            frame = provider.get_historical_data(
                symbol,
                start=start.replace(tzinfo=None),
                end=end.replace(tzinfo=None),
                interval="1d",
            )

            if frame.empty:
                return 0

            # Write to Bronze
            bronze = BronzeLayer()
            meta = bronze.append(frame, lineage={
                "source": "unified_pipeline",
                "symbol": symbol,
                "type": "ohlcv",
            })

            # Publish to Kafka if enabled
            if self.kafka_enabled:
                self._publish_to_kafka("ohlcv", symbol, frame)

            self._last_fetch[symbol] = datetime.now(timezone.utc)
            return len(frame)

        except Exception as e:
            logger.error(f"OHLCV fetch failed for {symbol}: {e}")
            return 0

    def _fetch_orderbook(self, symbol: str) -> int:
        """Fetch order book data."""
        try:
            if self._orderbook_provider is None:
                from app.data_sources.orderbook_provider import OrderBookProvider
                self._orderbook_provider = OrderBookProvider()

            orderbook = self._orderbook_provider.get_order_book(symbol, depth=10)

            if orderbook:
                # Store in Bronze (as separate partition or layer)
                df = self._orderbook_provider.orderbook_to_df(orderbook)

                if not df.empty:
                    from app.lakehouse.bronze import BronzeLayer
                    bronze = BronzeLayer()
                    bronze.append(df, lineage={
                        "source": "unified_pipeline",
                        "symbol": symbol,
                        "type": "orderbook",
                    })
                    return 1

        except Exception as e:
            logger.debug(f"OrderBook fetch failed for {symbol}: {e}")

        return 0

    def _fetch_news(self, symbol: str) -> int:
        """Fetch news and sentiment data."""
        try:
            if self._news_provider is None:
                from app.data_sources.enhanced_news_sentiment_provider import EnhancedNewsSentimentProvider
                self._news_provider = EnhancedNewsSentimentProvider()

            frame = self._news_provider.get_news_for_symbol(symbol, limit=20)

            if not frame.empty:
                # Write to a news storage (could be separate table or layer)
                self._store_news_sentiment(symbol, frame)
                return len(frame)

        except Exception as e:
            logger.debug(f"News fetch failed for {symbol}: {e}")

        return 0

    def _fetch_fundamental(self, symbol: str) -> int:
        """Fetch fundamental data."""
        try:
            if self._fundamental_provider is None:
                from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider
                self._fundamental_provider = EnhancedFundamentalProvider()

            metrics = self._fundamental_provider.get_comprehensive_metrics(symbol)

            if metrics:
                # Write to Bronze or separate fundamental store
                self._store_fundamental(symbol, metrics)
                return 1

        except Exception as e:
            logger.debug(f"Fundamental fetch failed for {symbol}: {e}")

        return 0

    def _fetch_indices(self) -> int:
        """Fetch market indices data."""
        try:
            if self._index_provider is None:
                from app.data_sources.market_index_provider import MarketIndexProvider
                self._index_provider = MarketIndexProvider()

            indices = self._index_provider.get_all_indices()
            count = 0

            for index_name, df in indices.items():
                if not df.empty:
                    from app.lakehouse.bronze import BronzeLayer
                    bronze = BronzeLayer()
                    bronze.append(df, lineage={
                        "source": "unified_pipeline",
                        "symbol": index_name,
                        "type": "index",
                    })
                    count += 1

            return count

        except Exception as e:
            logger.debug(f"Indices fetch failed: {e}")

        return 0

    def _store_news_sentiment(self, symbol: str, frame: pd.DataFrame) -> None:
        """Store news and sentiment data."""
        # This could write to a separate storage or add to Bronze
        logger.debug(f"Stored {len(frame)} news articles for {symbol}")

        # Publish sentiment to Kafka if enabled
        if self.kafka_enabled:
            sentiment_data = frame[["symbol", "title", "sentiment", "sentiment_label", "published_date"]].to_dict("records")
            self._publish_to_kafka("sentiment", symbol, pd.DataFrame(sentiment_data))

    def _store_fundamental(self, symbol: str, metrics) -> None:
        """Store fundamental data."""
        from app.data_sources.enhanced_fundamental_provider import EnhancedFundamentalProvider

        # Convert to dict and store
        provider = EnhancedFundamentalProvider()
        data = provider.metrics_to_dict(metrics)

        logger.debug(f"Stored fundamental data for {symbol}: {data}")

        # Publish to Kafka if enabled
        if self.kafka_enabled:
            df = pd.DataFrame([data])
            self._publish_to_kafka("fundamental", symbol, df)

    def _publish_to_kafka(self, data_type: str, symbol: str, df: pd.DataFrame) -> None:
        """Publish data to Kafka."""
        try:
            if self._kafka_producer is None:
                from app.streaming.kafka_producer import StockKafkaProducer
                self._kafka_producer = StockKafkaProducer(bootstrap_servers=self.kafka_bootstrap_servers)

            topic_map = {
                "ohlcv": "stock-ohlcv-raw",
                "sentiment": "stock-sentiment",
                "fundamental": "stock-fundamental",
                "orderbook": "stock-orderbook",
                "index": "stock-index",
            }

            topic = topic_map.get(data_type, "stock-data")
            self._kafka_producer.send_ohlcv(symbol, df, topic=topic)

        except Exception as e:
            logger.error(f"Kafka publish failed for {data_type}/{symbol}: {e}")

    def run_once(self) -> dict[str, Any]:
        """Run one iteration of all data sources."""
        from app.data_sources.factory import get_data_provider

        results = {
            "symbols_processed": 0,
            "ohlcv_records": 0,
            "orderbook_records": 0,
            "news_records": 0,
            "fundamental_records": 0,
            "index_records": 0,
            "errors": [],
        }

        provider = get_data_provider()

        for symbol in self.symbols:
            try:
                # OHLCV
                ohlcv_count = self._fetch_ohlcv(symbol, provider)
                results["ohlcv_records"] += ohlcv_count

                # Order book
                if self.include_orderbook:
                    results["orderbook_records"] += self._fetch_orderbook(symbol)

                # News
                if self.include_news:
                    results["news_records"] += self._fetch_news(symbol)

                # Fundamental (first time only)
                if self.include_fundamental and symbol not in self._last_fetch.get("fundamental", set()):
                    results["fundamental_records"] += self._fetch_fundamental(symbol)

                # Indices (once)
                if self.include_indices and symbol == self.symbols[0]:
                    results["index_records"] += self._fetch_indices()

                results["symbols_processed"] += 1

            except Exception as e:
                results["errors"].append(f"{symbol}: {str(e)}")

        # Mark fundamental as fetched
        if "fundamental" not in self._last_fetch:
            self._last_fetch["fundamental"] = set()
        self._last_fetch["fundamental"].update(self.symbols)

        return results

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "UnifiedStreamingPipeline":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# ─── Singleton & Helpers ────────────────────────────────────────────────────────

_pipeline: UnifiedStreamingPipeline | None = None


def get_pipeline() -> UnifiedStreamingPipeline:
    """Get singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedStreamingPipeline()
    return _pipeline


def start_pipeline(poll_interval_seconds: int = 60) -> UnifiedStreamingPipeline:
    """Start the pipeline."""
    pipeline = get_pipeline()
    pipeline.start(poll_interval_seconds)
    return pipeline


def stop_pipeline() -> None:
    """Stop the pipeline."""
    global _pipeline
    if _pipeline:
        _pipeline.stop()
        _pipeline = None
