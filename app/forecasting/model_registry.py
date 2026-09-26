"""Filesystem model registry used even when MySQL is unavailable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.forecasting.arima import ARIMAForecastModel
from app.forecasting.base import BaseForecastModel
from app.forecasting.linear_regression import LinearRegressionForecastModel

MODEL_CLASSES: dict[str, type[BaseForecastModel]] = {
    "linear_regression": LinearRegressionForecastModel,
    "arima": ARIMAForecastModel,
}

def _lazy_load_lstm():
    """Lazy load LSTM model to avoid PyTorch DLL issues on Windows."""
    try:
        from app.forecasting.lstm import LSTMForecastModel
        return LSTMForecastModel
    except OSError:
        return None


def model_directory(symbol: str, model_name: str) -> Path:
    path = settings.model_dir_path / symbol.upper() / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_registry_record(symbol: str, model_name: str, record: dict[str, Any]) -> Path:
    directory = model_directory(symbol, model_name)
    path = directory / "registry.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def load_registry_record(symbol: str, model_name: str) -> dict[str, Any]:
    path = model_directory(symbol, model_name) / "registry.json"
    if not path.exists():
        raise NotFoundError(f"No trained model found for {symbol} / {model_name}. Please train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(symbol: str, model_name: str) -> BaseForecastModel:
    if model_name == "lstm":
        lstm_cls = _lazy_load_lstm()
        if lstm_cls is None:
            raise NotFoundError("LSTM model unavailable due to PyTorch initialization error on Windows.")
        MODEL_CLASSES["lstm"] = lstm_cls
    if model_name not in MODEL_CLASSES:
        raise NotFoundError(f"Unknown model: {model_name}")
    directory = model_directory(symbol, model_name)
    marker = directory / "registry.json"
    if not marker.exists():
        raise NotFoundError(f"No trained model found for {symbol} / {model_name}. Please train the model first.")
    return MODEL_CLASSES[model_name].load(directory)


def list_trained_models(symbol: str) -> list[dict[str, Any]]:
    root = settings.model_dir_path / symbol.upper()
    if not root.exists():
        return []
    records = []
    for child in root.iterdir():
        registry = child / "registry.json"
        if registry.exists():
            try:
                records.append(json.loads(registry.read_text(encoding="utf-8")))
            except Exception:
                pass
    return records
