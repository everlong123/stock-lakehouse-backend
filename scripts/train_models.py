"""Train Linear Regression, ARIMA, and LSTM for selected symbols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging_config import get_logger
from app.forecasting.trainer import train_model

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasting models.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--models", default="linear_regression,arima,lstm")
    parser.add_argument("--epochs", type=int, default=8, help="LSTM epochs for a faster local demo.")
    args = parser.parse_args()
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        extra = {"epochs": args.epochs} if model_name == "lstm" else {}
        result = train_model(args.symbol.upper(), model_name, extra_params=extra)
        logger.info(
            "%s %s MAE=%.4f RMSE=%.4f MAPE=%.4f DA=%.4f",
            args.symbol,
            model_name,
            result["mae"],
            result["rmse"],
            result["mape"],
            result["directional_accuracy"],
        )


if __name__ == "__main__":
    main()
