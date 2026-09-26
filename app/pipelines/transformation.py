"""Bronze to Silver transformation step."""

from __future__ import annotations

from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.lakehouse.bronze import BronzeLayer
from app.lakehouse.silver import SilverLayer

logger = get_logger(__name__)


def transform_to_silver(symbol: str) -> dict:
    """Read Bronze, clean it, and write Silver."""
    bronze_frame = BronzeLayer().read(symbol)
    if bronze_frame.empty:
        raise DataValidationError(f"No Bronze data found for {symbol}. Run ingestion first.")
    silver = SilverLayer()
    silver_frame, quality = silver.transform(bronze_frame)
    write_info = silver.write(silver_frame, quality)
    logger.info("Silver transform complete for %s: %s", symbol, quality)
    return {"quality": quality, "write": write_info}
