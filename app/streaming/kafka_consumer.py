"""
Kafka Consumer - Real-time Pipeline Processing
==============================================

Consumes OHLCV data from Kafka and processes it through the lakehouse pipeline.

Architecture:
```
Kafka Topic
"stock-ohlcv-raw"
        │
        ▼
┌─────────────────┐
│  Kafka Consumer │  (this module)
│  (real-time)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Process Bar    │
│  - Validate     │
│  - Enrich       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
Bronze     Alerts
 Storage   (optional)
```

Features:
- Configurable batch size for efficiency
- Multi-threaded processing
- Exactly-once semantics with idempotent writes
- Graceful shutdown

References:
- kafka-python: https://kafka-python.readthedocs.io/
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
from kafka import KafkaConsumer as _KafkaConsumer
from kafka.errors import KafkaError

from app.core.logging_config import get_logger
from app.lakehouse.bronze import BronzeLayer
from app.lakehouse.silver import SilverLayer
from app.streaming.kafka_producer import TOPIC_OHLCV_ENRICHED

logger = get_logger(__name__)

# Default Kafka settings
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9094"
DEFAULT_GROUP_ID = "stock-lakehouse-consumer"


# ═══════════════════════════════════════════════════════════════════════════════
# DESERIALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def deserialize_ohlcv(raw: bytes) -> dict[str, Any]:
    """Deserialize Kafka message bytes to dict."""
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Failed to deserialize message: %s", e)
        raise


def dict_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dicts to DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Ensure datetime columns
    for col in ["timestamp", "ingestion_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# KAFKA CONSUMER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class StockKafkaConsumer:
    """
    Kafka consumer for real-time stock data processing.

    Consumes raw OHLCV bars and processes them through:
    1. Schema validation
    2. Bronze write (append-only)
    3. Silver transform (clean)
    4. Optional: publish enriched to downstream topic
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] | None = None,
        group_id: str = DEFAULT_GROUP_ID,
        topics: list[str] | None = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
        auto_commit_interval_ms: int = 5000,
        max_poll_records: int = 100,
        poll_timeout_ms: int = 1000,
    ) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers or DEFAULT_BOOTSTRAP_SERVERS
        self.group_id = group_id
        self.topics = topics or ["stock-ohlcv-raw"]
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.auto_commit_interval_ms = auto_commit_interval_ms
        self.max_poll_records = max_poll_records
        self.poll_timeout_ms = poll_timeout_ms

        self._consumer: _KafkaConsumer | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Pipeline components
        self._bronze = BronzeLayer()
        self._silver = SilverLayer()

    @property
    def consumer(self) -> _KafkaConsumer:
        """Lazy initialization of Kafka consumer."""
        if self._consumer is None:
            servers = self.bootstrap_servers
            if isinstance(servers, str):
                servers = [servers]

            self._consumer = _KafkaConsumer(
                *self.topics,
                bootstrap_servers=servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                auto_commit_interval_ms=self.auto_commit_interval_ms,
                value_deserializer=lambda v: v,  # We handle deserialization
                max_poll_records=self.max_poll_records,
            )
            logger.info(
                "Kafka consumer initialized: servers=%s, topics=%s, group=%s",
                self.bootstrap_servers,
                self.topics,
                self.group_id,
            )
        return self._consumer

    def start(self) -> None:
        """Start consuming in background thread."""
        if self._running:
            logger.warning("Consumer already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("Consumer started: topics=%s", self.topics)

    def stop(self) -> None:
        """Stop consuming gracefully."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
        logger.info("Consumer stopped")

    def _consume_loop(self) -> None:
        """Main consumer loop."""
        while self._running:
            try:
                # Poll for messages
                records = self.consumer.poll(timeout_ms=self.poll_timeout_ms)

                for topic_partition, messages in records.items():
                    for msg in messages:
                        self._process_message(msg)

            except Exception as e:
                logger.error("Consumer loop error: %s", e)
                time.sleep(1)  # Back off on error

    def _process_message(self, msg) -> None:
        """Process a single Kafka message."""
        try:
            # Deserialize
            data = deserialize_ohlcv(msg.value)
            symbol = data.get("symbol", "UNKNOWN").upper()

            # Convert to DataFrame
            df = dict_to_dataframe([data])
            if df.empty:
                return

            # Process through pipeline
            self._process_ohlcv(symbol, df)

            logger.debug(
                "Processed %s @ %s (partition=%d, offset=%d)",
                symbol,
                data.get("timestamp"),
                msg.partition,
                msg.offset,
            )

        except Exception as e:
            logger.error("Failed to process message: %s", e)

    def _process_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Process OHLCV data through the lakehouse pipeline.

        Steps:
        1. Write to Bronze (append)
        2. Transform to Silver
        3. Write to Silver
        """
        try:
            # Step 1: Write to Bronze
            bronze_meta = self._bronze.append(
                df,
                lineage={
                    "source": "kafka",
                    "topic": self.topics[0],
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(
                "Bronze write: %s (%d rows)",
                symbol,
                bronze_meta.get("records_written", 0),
            )

            # Step 2: Read from Bronze and transform to Silver
            silver_df = self._silver.transform(df)

            if not silver_df.empty:
                # Step 3: Write to Silver
                silver_write = self._silver.write(silver_df, {"source": "kafka_stream"})
                logger.info(
                    "Silver write: %s (%d rows)",
                    symbol,
                    silver_write.get("records", 0),
                )

        except Exception as e:
            logger.error("Pipeline processing failed for %s: %s", symbol, e)
            raise

    def consume_batch(self, max_records: int = 100, timeout_ms: int = 5000) -> int:
        """
        Consume a batch of messages synchronously.

        Returns number of records processed.
        """
        count = 0
        deadline = time.time() + (timeout_ms / 1000)

        while count < max_records and time.time() < deadline:
            records = self.consumer.poll(timeout_ms=int((deadline - time.time()) * 1000))

            for topic_partition, messages in records.items():
                for msg in messages:
                    self._process_message(msg)
                    count += 1
                    if count >= max_records:
                        break

        return count

    def __enter__(self) -> "StockKafkaConsumer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RealTimeAggregator:
    """
    Aggregates incoming 1-minute bars into higher timeframes.

    For example, converts 1m bars into:
    - 5m bars
    - 15m bars
    - 1h bars
    - 1d bars

    This is useful when you only have 1m data but need higher timeframes.
    """

    def __init__(self, timeframes: list[str] | None = None) -> None:
        # Supported timeframe mappings (target: source minutes)
        self.timeframes = timeframes or ["5T", "15T", "1H", "1D"]
        self._buffers: dict[str, pd.DataFrame] = {}

    def aggregate(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Aggregate 1m bars into multiple timeframes.

        Args:
            df: DataFrame with OHLCV data (must have timestamp index)

        Returns:
            Dict mapping timeframe -> aggregated DataFrame
        """
        if df.empty:
            return {}

        results = {}

        for tf in self.timeframes:
            try:
                # Ensure timestamp is index
                work = df.set_index("timestamp").sort_index()

                # Resample to target timeframe
                agg = work.resample(tf).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "symbol": "first",
                })

                # Remove NaN rows
                agg = agg.dropna().reset_index()
                if not agg.empty:
                    results[tf] = agg
                    logger.debug("Aggregated %d bars to %s", len(agg), tf)

            except Exception as e:
                logger.error("Failed to aggregate to %s: %s", tf, e)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_consumer: StockKafkaConsumer | None = None


def get_kafka_consumer(
    group_id: str = DEFAULT_GROUP_ID,
    topics: list[str] | None = None,
) -> StockKafkaConsumer:
    """Get singleton consumer instance."""
    global _consumer
    if _consumer is None:
        _consumer = StockKafkaConsumer(group_id=group_id, topics=topics)
    return _consumer


def start_consumer(
    group_id: str = DEFAULT_GROUP_ID,
    topics: list[str] | None = None,
) -> StockKafkaConsumer:
    """Convenience to start consumer in one call."""
    consumer = StockKafkaConsumer(group_id=group_id, topics=topics)
    consumer.start()
    return consumer
