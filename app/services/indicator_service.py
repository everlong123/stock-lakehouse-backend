"""Indicator service wrapping Pandas implementations."""

from __future__ import annotations

from app.indicators.service import IndicatorConfig, add_indicators, describe_indicators
from app.services.market_service import MarketService


class IndicatorService:
    def __init__(self) -> None:
        self.market = MarketService()

    def calculate(
        self,
        symbol: str,
        sma: bool = True,
        ema: bool = True,
        rsi: bool = True,
        macd: bool = True,
        bollinger: bool = True,
        sma_windows: list[int] | None = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_window: int = 20,
        bb_std: float = 2.0,
    ) -> dict:
        frame = self.market.get_history(symbol)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        windows = tuple(sma_windows or [5, 10, 20, 50])
        config = IndicatorConfig(
            sma_windows=windows,
            ema_windows=(12, 26),
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_window=bb_window,
            bb_std=bb_std,
        )
        enriched = add_indicators(frame, config)
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        if sma:
            keep.extend([col for col in enriched.columns if col.startswith("sma_")])
        if ema:
            keep.extend([col for col in enriched.columns if col.startswith("ema_")])
        if rsi:
            keep.append("rsi_14")
        if macd:
            keep.extend(["macd", "macd_signal", "macd_hist"])
        if bollinger:
            keep.extend(["bb_middle", "bb_upper", "bb_lower"])
        keep = [col for col in dict.fromkeys(keep) if col in enriched.columns]
        chart = enriched[keep].tail(400)
        latest = enriched.iloc[-1]
        summary = describe_indicators(latest)
        points = []
        for row in chart.itertuples(index=False):
            item = {col: (str(getattr(row, col)) if col == "timestamp" else _to_float(getattr(row, col))) for col in keep}
            points.append(item)
        return {
            "symbol": symbol.upper(),
            "latest": {key: _to_float(latest[key]) for key in keep if key != "timestamp" and key in latest},
            "summary": summary,
            "series": points,
            "disclaimer": "Indicator summary describes historical statistics only and is not an investment order.",
        }


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if number != number:  # NaN
            return None
        return number
    except Exception:
        return None
