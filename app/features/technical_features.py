"""Technical indicator features for the Gold layer."""

from __future__ import annotations

import pandas as pd

from app.indicators.service import IndicatorConfig, add_indicators


def add_technical_features(frame: pd.DataFrame, config: IndicatorConfig | None = None) -> pd.DataFrame:
    """Attach SMA/EMA/RSI/MACD/Bollinger features."""
    return add_indicators(frame, config=config)
