"""Feature engineering leakage tests."""

import pandas as pd

from app.features.feature_engineering import build_gold_features
from tests.conftest import make_ohlcv


def test_targets_are_shifted_forward() -> None:
    frame = make_ohlcv(80)
    gold = build_gold_features(frame)
    for index in range(len(gold) - 1):
        assert gold.loc[index, "target_close_next"] == gold.loc[index + 1, "close"]
    assert pd.isna(gold.loc[len(gold) - 1, "target_close_next"])


def test_lags_use_past_only() -> None:
    frame = make_ohlcv(80)
    gold = build_gold_features(frame)
    assert gold.loc[2, "close_lag_1"] == gold.loc[1, "close"]
    assert gold.loc[2, "close_lag_2"] == gold.loc[0, "close"]


def test_required_gold_columns_exist() -> None:
    gold = build_gold_features(make_ohlcv(120))
    expected = {
        "return",
        "log_return",
        "sma_20",
        "ema_12",
        "rsi_14",
        "macd",
        "bb_middle",
        "target_close_next",
        "target_return_next",
        "target_direction_next",
    }
    assert expected.issubset(gold.columns)
