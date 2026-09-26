"""Strategy package."""

from app.backtesting.strategies.ma_crossover import MACrossoverStrategy
from app.backtesting.strategies.rsi_strategy import RSIStrategy

STRATEGY_MAP = {
    "ma_crossover": MACrossoverStrategy,
    "rsi_strategy": RSIStrategy,
}

__all__ = ["MACrossoverStrategy", "RSIStrategy", "STRATEGY_MAP"]
