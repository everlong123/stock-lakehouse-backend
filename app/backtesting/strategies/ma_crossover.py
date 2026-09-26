"""Moving-average crossover strategy."""

from __future__ import annotations

import pandas as pd

from app.backtesting.strategy_base import BaseStrategy
from app.indicators.sma import sma


class MACrossoverStrategy(BaseStrategy):
    """Buy when short SMA crosses above long SMA. Sell on the opposite cross."""

    name = "ma_crossover"

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        short_window = int(self.parameters.get("short_window", 20))
        long_window = int(self.parameters.get("long_window", 50))
        close = frame["close"]
        short_ma = sma(close, short_window)
        long_ma = sma(close, long_window)
        position = (short_ma > long_ma).astype(int)
        signal = position.diff().fillna(0)
        # +1 golden cross, -1 death cross
        return signal.clip(-1, 1)
