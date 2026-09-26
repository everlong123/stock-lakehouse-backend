"""High-level training orchestrator used by the API and scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.constants import LINEAR_REGRESSION_FEATURES, LSTM_FEATURE_COLUMNS
from app.core.exceptions import ModelTrainingError
from app.core.logging_config import get_logger
from app.core.seeding import set_global_seed
from app.core.config import settings
from app.forecasting.arima import ARIMAForecastModel
from app.forecasting.linear_regression import LinearRegressionForecastModel
from app.forecasting.lstm import LSTMForecastModel
from app.forecasting.model_registry import MODEL_CLASSES, model_directory, save_registry_record
from app.forecasting.preprocessing import chronological_split
from app.lakehouse.gold import GoldLayer

logger = get_logger(__name__)


def _prepare_supervised(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    needed = feature_cols + ["target_close_next", "timestamp", "close"]
    present = [col for col in needed if col in frame.columns]
    clean = frame.loc[:, ~frame.columns.duplicated()][present].dropna(subset=feature_cols + ["target_close_next"]).copy()
    if clean.empty:
        raise ModelTrainingError("No supervised rows remain after dropping indicator warmup and target NaNs.")
    return clean


def train_model(
    symbol: str,
    model_name: str,
    horizon: int = 1,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one model on Gold data using a chronological split."""
    set_global_seed(settings.random_seed)
    extra_params = extra_params or {}
    gold = GoldLayer().read(symbol)
    if gold.empty:
        raise ModelTrainingError(f"No Gold data for {symbol}. Run the lakehouse pipeline first.")

    model_name = model_name.lower()
    if model_name not in MODEL_CLASSES:
        raise ModelTrainingError(f"Unsupported model: {model_name}")

    if model_name == "arima":
        clean = gold.dropna(subset=["close"]).sort_values("timestamp")
        split = chronological_split(clean)
        model = ARIMAForecastModel(order=tuple(extra_params.get("order", settings.arima_order_tuple)))
        model.fit(split.train, split.train["close"])
        metrics = model.evaluate(split.test, split.test["close"])
        feature_cols = ["close"]
        aligned_pred = model.predict(split.test)
        comparison = pd.DataFrame(
            {
                "timestamp": split.test["timestamp"].tolist()[: len(aligned_pred)],
                "actual": split.test["close"].tolist()[: len(aligned_pred)],
                "predicted": aligned_pred.tolist(),
            }
        )
    elif model_name == "lstm":
        feature_cols = [col for col in LSTM_FEATURE_COLUMNS if col in gold.columns]
        clean = _prepare_supervised(gold, feature_cols)
        split = chronological_split(clean)
        model = LSTMForecastModel(
            sequence_length=int(extra_params.get("sequence_length", settings.lstm_sequence_length)),
            hidden_size=int(extra_params.get("hidden_size", settings.lstm_hidden_size)),
            num_layers=int(extra_params.get("num_layers", settings.lstm_num_layers)),
            epochs=int(extra_params.get("epochs", settings.lstm_epochs)),
        )
        model.fit(split.train[feature_cols], split.train["target_close_next"], split.validation[feature_cols], split.validation["target_close_next"])
        metrics = model.evaluate(split.test[feature_cols], split.test["target_close_next"])
        predicted = model.predict(split.test[feature_cols])
        actual = split.test["target_close_next"].to_numpy()[model.sequence_length - 1 :]
        timestamps = split.test["timestamp"].to_numpy()[model.sequence_length - 1 :]
        n = min(len(predicted), len(actual), len(timestamps))
        comparison = pd.DataFrame(
            {
                "timestamp": list(timestamps[:n]),
                "actual": list(actual[:n]),
                "predicted": list(predicted[:n]),
            }
        )
    else:
        feature_cols = [col for col in LINEAR_REGRESSION_FEATURES if col in gold.columns]
        clean = _prepare_supervised(gold, feature_cols)
        split = chronological_split(clean)
        model = LinearRegressionForecastModel()
        model.fit(split.train[feature_cols], split.train["target_close_next"])
        metrics = model.evaluate(split.test[feature_cols], split.test["target_close_next"])
        predicted = model.predict(split.test[feature_cols])
        comparison = pd.DataFrame(
            {
                "timestamp": split.test["timestamp"].tolist(),
                "actual": split.test["target_close_next"].tolist(),
                "predicted": predicted.tolist(),
            }
        )

    directory = model_directory(symbol, model_name)
    model.save(directory)
    comparison_path = directory / "predictions.parquet"
    comparison.to_parquet(comparison_path, index=False)

    record = {
        "symbol": symbol.upper(),
        "model_name": model_name,
        "train_start": split.train["timestamp"].min(),
        "train_end": split.train["timestamp"].max(),
        "validation_start": split.validation["timestamp"].min(),
        "validation_end": split.validation["timestamp"].max(),
        "test_start": split.test["timestamp"].min(),
        "test_end": split.test["timestamp"].max(),
        "features": feature_cols,
        "parameters": {**model.metadata, "horizon": horizon, **extra_params},
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "directional_accuracy": metrics["directional_accuracy"],
        "model_path": str(directory),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Metrics are computed on a chronological hold-out split. They do not guarantee future performance.",
    }
    save_registry_record(symbol, model_name, record)
    logger.info("Trained %s for %s: %s", model_name, symbol, metrics)
    return {
        **record,
        "predictions": [
            {
                "timestamp": str(row.timestamp),
                "actual": float(row.actual),
                "predicted": float(row.predicted),
            }
            for row in comparison.itertuples(index=False)
        ],
    }
