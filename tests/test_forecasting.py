"""Forecasting tests using a chronological split and offline data."""

import pandas as pd

from app.features.feature_engineering import build_gold_features
from app.forecasting.linear_regression import LinearRegressionForecastModel
from app.forecasting.preprocessing import chronological_split
from tests.conftest import make_ohlcv


def test_chronological_split_order() -> None:
    frame = make_ohlcv(100)
    split = chronological_split(frame)
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_linear_regression_fit_predict() -> None:
    gold = build_gold_features(make_ohlcv(220))
    features = ["close", "volume", "return", "sma_20", "rsi_14", "close_lag_1"]
    clean = gold.dropna(subset=features + ["target_close_next"])
    split = chronological_split(clean)
    model = LinearRegressionForecastModel()
    model.fit(split.train[features], split.train["target_close_next"])
    preds = model.predict(split.test[features])
    assert len(preds) == len(split.test)
    metrics = model.evaluate(split.test[features], split.test["target_close_next"])
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
