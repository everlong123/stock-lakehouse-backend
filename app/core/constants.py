"""Shared constants for the Stock Lakehouse platform."""

from __future__ import annotations

SUPPORTED_SYMBOLS: list[str] = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
SUPPORTED_INTERVALS: list[str] = ["1d", "1h", "15m", "5m"]

OHLCV_COLUMNS: list[str] = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "ingestion_time",
]

BRONZE_PARTITION_COLUMNS: list[str] = ["symbol", "year", "month"]

GOLD_FEATURE_COLUMNS: list[str] = [
    "return",
    "log_return",
    "price_change",
    "volume_change",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "rolling_std_20",
    "rolling_min_20",
    "rolling_max_20",
    "volume_ma_20",
    "close_lag_1",
    "close_lag_2",
    "return_lag_1",
    "volume_lag_1",
]

GOLD_TARGET_COLUMNS: list[str] = [
    "target_close_next",
    "target_return_next",
    "target_direction_next",
]

LINEAR_REGRESSION_FEATURES: list[str] = [
    "close",
    "volume",
    "return",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "rolling_std_20",
    "close_lag_1",
    "close_lag_2",
    "return_lag_1",
    "volume_lag_1",
]

LSTM_FEATURE_COLUMNS: list[str] = [
    "close",
    "volume",
    "return",
    "sma_20",
    "ema_12",
    "rsi_14",
    "macd",
    "bb_middle",
    "rolling_std_20",
]

ACADEMIC_DISCLAIMER = (
    "Kết quả phân tích và dự báo chỉ phục vụ nghiên cứu học thuật, "
    "không phải khuyến nghị đầu tư và không cam kết lợi nhuận tương lai."
)

INTERVAL_TO_PANDAS_FREQ: dict[str, str] = {
    "1d": "B",
    "1h": "h",
    "15m": "15min",
    "5m": "5min",
}

INTERVAL_BARS_PER_YEAR: dict[str, int] = {
    "1d": 252,
    "1h": 252 * 6.5,
    "15m": 252 * 26,
    "5m": 252 * 78,
}
