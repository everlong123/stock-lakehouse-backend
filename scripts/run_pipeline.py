"""Run the Bronze → Silver → Gold pipeline for one or more symbols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.constants import SUPPORTED_SYMBOLS
from app.core.logging_config import get_logger
from app.pipelines.pipeline_runner import run_symbol_pipeline

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock lakehouse pipeline.")
    parser.add_argument("--symbol", default="ALL", help="Ticker or ALL")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--source", default=None, help="sample or yfinance")
    args = parser.parse_args()
    symbols = SUPPORTED_SYMBOLS if args.symbol.upper() == "ALL" else [args.symbol.upper()]
    for symbol in symbols:
        result = run_symbol_pipeline(symbol=symbol, interval=args.interval, source_name=args.source)
        logger.info("Finished %s status=%s records=%s", symbol, result["status"], result["records_processed"])


if __name__ == "__main__":
    main()
