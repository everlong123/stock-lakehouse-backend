"""Indicator unit tests."""

import numpy as np
import pandas as pd

from app.indicators.bollinger import bollinger_bands
from app.indicators.ema import ema
from app.indicators.macd import macd
from app.indicators.rsi import rsi
from app.indicators.sma import sma


def test_sma_known_values() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, 3)
    assert np.isclose(result.iloc[2], 2.0)
    assert np.isclose(result.iloc[4], 4.0)
    assert pd.isna(result.iloc[1])


def test_ema_weights_recent_prices_more_than_sma() -> None:
    series = pd.Series([10.0] * 10 + [20.0] * 10)
    ema_values = ema(series, 5)
    sma_values = sma(series, 5)
    # During the step-up, EMA puts more weight on recent closes than SMA.
    assert ema_values.iloc[12] > sma_values.iloc[12]
    assert ema_values.notna().sum() > 0


def test_rsi_bounds() -> None:
    series = pd.Series(np.linspace(100, 140, 40))
    values = rsi(series, 14)
    valid = values.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()
    assert valid.iloc[-1] > 50


def test_macd_columns() -> None:
    series = pd.Series(np.linspace(50, 80, 80))
    frame = macd(series)
    assert {"macd", "macd_signal", "macd_hist"} <= set(frame.columns)
    assert frame["macd_hist"].notna().sum() > 0


def test_bollinger_envelope() -> None:
    series = pd.Series(np.linspace(20, 40, 40) + np.sin(np.arange(40)))
    bands = bollinger_bands(series, 20, 2)
    valid = bands.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()
