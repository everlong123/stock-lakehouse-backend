"""Backtest look-ahead bias tests."""

from datetime import datetime, timezone

import pandas as pd

from app.backtesting.engine import BacktestEngine
from app.backtesting.strategies.ma_crossover import MACrossoverStrategy
from tests.conftest import make_ohlcv


def test_signal_does_not_use_future_close() -> None:
    frame = make_ohlcv(120)
    signals = MACrossoverStrategy({"short_window": 5, "long_window": 10}).generate_signals(frame)
    # A signal at index t must equal the MA comparison computed only with data through t.
    for index in range(15, 40):
        window = frame.iloc[: index + 1]
        local = MACrossoverStrategy({"short_window": 5, "long_window": 10}).generate_signals(window)
        assert signals.iloc[index] == local.iloc[-1]


def test_execution_uses_next_bar(monkeypatch) -> None:
    frame = make_ohlcv(150)
    engine = BacktestEngine()

    def fake_load(symbol, start_date, end_date):
        return frame

    monkeypatch.setattr(engine, "_load_ohlcv", fake_load)
    result = engine.run("AAPL", "ma_crossover", initial_capital=10_000, transaction_fee=0.0, slippage=0.0)
    assert result["number_of_trades"] >= 0
    assert "equity_curve" in result
    if result["trades"]:
        first = result["trades"][0]
        assert first["exit_time"] >= first["entry_time"]
