"""RSI threshold strategy."""

from __future__ import annotations

import pandas as pd

from app.backtesting.strategy_base import BaseStrategy
from app.indicators.rsi import rsi


class RSIStrategy(BaseStrategy):
    """Buy when RSI drops below the lower threshold. Sell when RSI exceeds the upper threshold."""

    name = "rsi_strategy"

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        period = int(self.parameters.get("rsi_period", 14))
        lower = float(self.parameters.get("lower_threshold", 30))
        upper = float(self.parameters.get("upper_threshold", 70))
        values = rsi(frame["close"], period)
        signal = pd.Series(0, index=frame.index, dtype=float)
        signal.loc[values < lower] = 1
        signal.loc[values > upper] = -1
        # Keep only threshold crossings to avoid repeated orders on every bar.
        previous = signal.replace(0, pd.NA).ffill().shift(1)
        crossings = signal.where(signal != previous, 0).fillna(0)
        return crossings.clip(-1, 1)
