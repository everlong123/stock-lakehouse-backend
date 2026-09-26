"""
Kafka Producer - Real-time Stock Data Ingestion
==============================================

Publishes OHLCV data to Kafka topics for real-time processing.

Architecture:
```
yfinance / Web Scraper
        │
        ▼
┌─────────────────┐
│  Kafka Topic    │  "stock-ohlcv-raw"
│  (raw OHLCV)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kafka Consumer │
│  (real-time)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
Bronze     Real-time
(Parquet)  Dashboard
```

Topics:
- stock-ohlcv-raw: Raw OHLCV bars
- stock-ohlcv-enriched: Enriched with indicators
- stock-alerts: Price alerts (optional)

References:
- Kafka Python: https://kafka-python.readthedocs.io/
- Confluent Kafka: https://docs.confluent.io/clients-confluent-kafka-python/
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import pandas as pd
from kafka import KafkaProducer as _KafkaProducer
from kafka.errors import KafkaError

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.data_sources.factory import get_data_provider

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Kafka topics
TOPIC_OHLCV_RAW = "stock-ohlcv-raw"
TOPIC_OHLCV_ENRICHED = "stock-ohlcv-enriched"
TOPIC_ALERTS = "stock-alerts"

# Default broker
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9094"


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZERS
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_ohlcv(row: pd.Series) -> bytes:
    """Serialize a single OHLCV row to JSON bytes for Kafka."""
    data = {
        "symbol": str(row.get("symbol", "")).upper(),
        "timestamp": _ensure_iso(row.get("timestamp")),
        "open": float(row.get("open", 0)),
        "high": float(row.get("high", 0)),
        "low": float(row.get("low", 0)),
        "close": float(row.get("close", 0)),
        "adj_close": float(row.get("adj_close", 0)) if pd.notna(row.get("adj_close")) else None,
        "volume": float(row.get("volume", 0)),
        "source": str(row.get("source", "unknown")),
        "ingestion_time": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(data, default=str).encode("utf-8")


def serialize_alert(symbol: str, price: float, condition: str, message: str) -> bytes:
    """Serialize an alert message."""
    data = {
        "type": "alert",
        "symbol": symbol.upper(),
        "price": price,
        "condition": condition,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(data, default=str).encode("utf-8")


def _ensure_iso(value: Any) -> str:
    """Ensure value is ISO format string."""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# ═══════════════════════════════════════════════════════════════════════════════
# KAFKA PRODUCER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class StockKafkaProducer:
    """
    Kafka producer for publishing stock OHLCV data.

    Features:
    - Async send with callbacks
    - Automatic topic creation
    - Compression for efficiency
    - Configurable acks/retries
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] | None = None,
        compression_type: str = "gzip",
        acks: str = "all",
        retries: int = 3,
        linger_ms: int = 10,
    ) -> None:
        settings = get_settings()
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers or DEFAULT_BOOTSTRAP_SERVERS
        self._producer: _KafkaProducer | None = None
        self._compression = compression_type
        self._acks = acks
        self._retries = retries
        self._linger_ms = linger_ms
        self._closed = False

    @property
    def producer(self) -> _KafkaProducer:
        """Lazy initialization of Kafka producer."""
        if self._producer is None:
            self._producer = _KafkaProducer(
                bootstrap_servers=self.bootstrap_servers
                    if isinstance(self.bootstrap_servers, list)
                    else [self.bootstrap_servers],
                value_serializer=lambda v: v,  # We handle serialization
                acks=self._acks,
                retries=self._retries,
                compression_type=self._compression,
                linger_ms=self._linger_ms,
                # Enable exactly-once semantics (idempotent producer)
                enable_idempotence=True,
            )
            logger.info("Kafka producer initialized: %s", self.bootstrap_servers)
        return self._producer

    def send_ohlcv(
        self,
        symbol: str,
        df: pd.DataFrame,
        topic: str = TOPIC_OHLCV_RAW,
        key: str | None = None,
    ) -> int:
        """
        Send OHLCV DataFrame to Kafka topic.

        Args:
            symbol: Ticker symbol for partition key
            df: DataFrame with OHLCV columns
            topic: Kafka topic name
            key: Optional message key (default: symbol)

        Returns:
            Number of messages sent
        """
        if df.empty:
            logger.warning("Empty DataFrame for %s, skipping", symbol)
            return 0

        count = 0
        key_bytes = (key or symbol).encode("utf-8")

        for _, row in df.iterrows():
            try:
                value = serialize_ohlcv(row)
                future = self.producer.send(
                    topic,
                    key=key_bytes,
                    value=value,
                )
                # Add callback for async logging
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)
                count += 1
            except Exception as e:
                logger.error("Failed to send row for %s: %s", symbol, e)

        # Flush to ensure all messages are sent
        self.producer.flush(timeout=5)
        logger.info("Sent %d OHLCV messages for %s to %s", count, symbol, topic)
        return count

    def send_ohlcv_stream(
        self,
        symbol: str,
        df: pd.DataFrame,
        topic: str = TOPIC_OHLCV_RAW,
    ) -> int:
        """Send all rows as a single batch message (more efficient for bulk)."""
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            records.append(json.loads(serialize_ohlcv(row).decode("utf-8")))

        value = json.dumps({"symbol": symbol.upper(), "bars": records}).encode("utf-8")
        key = symbol.encode("utf-8")

        try:
            self.producer.send(topic, key=key, value=value)
            self.producer.flush(timeout=10)
            logger.info("Sent batch of %d bars for %s", len(records), symbol)
            return len(records)
        except Exception as e:
            logger.error("Batch send failed for %s: %s", symbol, e)
            return 0

    def send_alert(
        self,
        symbol: str,
        price: float,
        condition: str,
        message: str,
        topic: str = TOPIC_ALERTS,
    ) -> bool:
        """Send a price alert to Kafka."""
        try:
            value = serialize_alert(symbol, price, condition, message)
            self.producer.send(topic, key=symbol.encode("utf-8"), value=value)
            self.producer.flush(timeout=5)
            logger.info("Alert sent for %s: %s", symbol, condition)
            return True
        except Exception as e:
            logger.error("Alert send failed for %s: %s", symbol, e)
            return False

    def _on_send_success(self, record_metadata) -> None:
        """Callback on successful send."""
        logger.debug(
            "Message delivered to %s [%d] @ offset %d",
            record_metadata.topic,
            record_metadata.partition,
            record_metadata.offset,
        )

    def _on_send_error(self, excp) -> None:
        """Callback on send error."""
        logger.error("Message delivery failed: %s", excp)

    def close(self) -> None:
        """Close the producer."""
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
            self._producer = None
            self._closed = True
            logger.info("Kafka producer closed")

    def __enter__(self) -> "StockKafkaProducer":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING INGESTER
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingIngester:
    """
    Continuous streaming ingester that polls data sources and publishes to Kafka.

    This replaces the batch run_pipeline.py for real-time scenarios.
    Polls at configurable intervals and sends only new data.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str = "1m",
        bootstrap_servers: str | None = None,
        source_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.symbols = symbols or settings.crawl_symbols.split(",")
        self.interval = interval
        self.source_name = source_name or settings.data_source
        self.producer = StockKafkaProducer(bootstrap_servers=bootstrap_servers)
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_timestamps: dict[str, datetime] = {}

    def start(self, poll_interval_seconds: int = 60) -> None:
        """
        Start streaming in background thread.

        Args:
            poll_interval_seconds: How often to poll data source
        """
        if self._running:
            logger.warning("Streaming ingester already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(poll_interval_seconds,),
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Streaming ingester started: symbols=%s, interval=%s, poll=%ds",
            self.symbols,
            self.interval,
            poll_interval_seconds,
        )

    def stop(self) -> None:
        """Stop the streaming ingester."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.producer.close()
        logger.info("Streaming ingester stopped")

    def _run_loop(self, poll_interval: int) -> None:
        """Main polling loop."""
        provider = get_data_provider(self.source_name)

        while self._running:
            try:
                for symbol in self.symbols:
                    symbol = symbol.strip().upper()
                    # Get latest data
                    df = provider.get_historical_data(
                        symbol=symbol,
                        interval=self.interval,
                        start=None,  # Get latest only
                        end=None,
                    )

                    if df.empty:
                        logger.debug("No new data for %s", symbol)
                        continue

                    # Filter to only new rows
                    last_ts = self._last_timestamps.get(symbol)
                    if last_ts is not None:
                        df = df[df["timestamp"] > last_ts]

                    if not df.empty:
                        count = self.producer.send_ohlcv(symbol, df)
                        self._last_timestamps[symbol] = df["timestamp"].max()
                        logger.info("Streamed %d bars for %s", count, symbol)

            except Exception as e:
                logger.error("Streaming error: %s", e)

            # Sleep between polls
            time.sleep(poll_interval)

    def ingest_once(self) -> dict[str, int]:
        """Run one iteration manually (for testing/scheduling)."""
        results = {}
        provider = get_data_provider(self.source_name)

        for symbol in self.symbols:
            symbol = symbol.strip().upper()
            try:
                df = provider.get_historical_data(
                    symbol=symbol,
                    interval=self.interval,
                )
                if not df.empty:
                    count = self.producer.send_ohlcv(symbol, df)
                    results[symbol] = count
            except Exception as e:
                logger.error("Failed to ingest %s: %s", symbol, e)
                results[symbol] = 0

        return results

    def __enter__(self) -> "StreamingIngester":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_producer: StockKafkaProducer | None = None


def get_kafka_producer() -> StockKafkaProducer:
    """Get singleton producer instance."""
    global _producer
    if _producer is None:
        _producer = StockKafkaProducer()
    return _producer


def publish_ohlcv(symbol: str, df: pd.DataFrame, topic: str = TOPIC_OHLCV_RAW) -> int:
    """Convenience function to publish OHLCV data."""
    return get_kafka_producer().send_ohlcv(symbol, df, topic)


def publish_alert(symbol: str, price: float, condition: str, message: str) -> bool:
    """Convenience function to publish an alert."""
    return get_kafka_producer().send_alert(symbol, price, condition, message)
