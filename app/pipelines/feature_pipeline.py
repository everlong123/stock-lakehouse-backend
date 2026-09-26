"""Silver to Gold feature pipeline."""

from __future__ import annotations

from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer

logger = get_logger(__name__)


def build_gold_layer(symbol: str) -> dict:
    """Read Silver, engineer features/targets, and write Gold."""
    silver_frame = SilverLayer().read(symbol)
    if silver_frame.empty:
        raise DataValidationError(f"No Silver data found for {symbol}. Transform Silver first.")
    gold = GoldLayer()
    gold_frame = gold.transform(silver_frame)
    write_info = gold.write(gold_frame)
    logger.info("Gold build complete for %s: %s rows", symbol, write_info["records"])
    return write_info
