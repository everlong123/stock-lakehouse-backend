"""ARIMA close-price forecast using statsmodels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from app.core.config import settings
from app.core.exceptions import ModelTrainingError, PredictionError
from app.core.logging_config import get_logger
from app.forecasting.base import BaseForecastModel
from app.forecasting.evaluator import evaluate_forecast

logger = get_logger(__name__)


class ARIMAForecastModel(BaseForecastModel):
    """Univariate ARIMA on the close series. Order is configurable."""

    model_name = "arima"

    def __init__(self, order: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        self.order = order or settings.arima_order_tuple
        self.result: ARIMAResults | None = None
        self.history: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "ARIMAForecastModel":
        target = np.asarray(y, dtype=float)
        if len(target) < 20:
            raise ModelTrainingError("ARIMA needs at least 20 close observations.")
        try:
            model = ARIMA(target, order=self.order)
            self.result = model.fit()
        except Exception as exc:
            raise ModelTrainingError(f"ARIMA fitting failed for order={self.order}: {exc}") from exc
        self.history = target
        self.is_fitted = True
        self.feature_names = ["close"]
        self.metadata = {"order": list(self.order), "n_samples": int(len(target))}
        logger.info("ARIMA fitted with order=%s on %s observations", self.order, len(target))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.result is None:
            raise PredictionError("ARIMA model is not trained.")
        horizon = len(X) if not isinstance(X, int) else int(X)
        forecast = self.result.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)

    def in_sample_predict(self) -> np.ndarray:
        if self.result is None:
            raise PredictionError("ARIMA model is not trained.")
        return np.asarray(self.result.fittedvalues, dtype=float)

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        previous = np.concatenate([[self.history[-1]], y_true[:-1]]) if len(y_true) else None
        return evaluate_forecast(y_true, y_pred, previous)

    def save(self, directory: Path) -> Path:
        if self.result is None:
            raise ModelTrainingError("Cannot save an unfitted ARIMA model.")
        directory.mkdir(parents=True, exist_ok=True)
        self.result.save(str(directory / "arima_result.pkl"))
        np.save(directory / "history.npy", self.history)
        (directory / "metadata.json").write_text(
            json.dumps({"order": list(self.order), "metadata": self.metadata}, indent=2),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ARIMAForecastModel":
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(order=tuple(payload.get("order", [5, 1, 0])))
        instance.result = ARIMAResults.load(str(directory / "arima_result.pkl"))
        instance.history = np.load(directory / "history.npy")
        instance.metadata = payload.get("metadata", {})
        instance.is_fitted = True
        instance.feature_names = ["close"]
        return instance
