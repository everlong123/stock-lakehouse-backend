"""Linear regression next-close model using scikit-learn."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelTrainingError, PredictionError
from app.core.logging_config import get_logger
from app.forecasting.base import BaseForecastModel
from app.forecasting.evaluator import evaluate_forecast

logger = get_logger(__name__)


class LinearRegressionForecastModel(BaseForecastModel):
    """Ordinary least squares on Gold features. Scaler is fit on train only."""

    model_name = "linear_regression"

    def __init__(self) -> None:
        super().__init__()
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LinearRegressionForecastModel":
        frame = pd.DataFrame(X)
        target = np.asarray(y, dtype=float)
        if frame.empty:
            raise ModelTrainingError("Linear regression received an empty training set.")
        self.feature_names = list(frame.columns)
        scaled = self.scaler.fit_transform(frame)
        self.model.fit(scaled, target)
        self.is_fitted = True
        self.metadata = {"n_features": len(self.feature_names), "n_samples": int(len(frame))}
        logger.info("Linear regression fitted on %s rows and %s features", len(frame), len(self.feature_names))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("Linear regression model is not trained.")
        frame = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        scaled = self.scaler.transform(frame)
        return np.asarray(self.model.predict(scaled), dtype=float)

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
        frame = pd.DataFrame(X)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(frame)
        previous = None
        if "close" in frame.columns:
            previous = np.asarray(frame["close"], dtype=float).reshape(len(frame), -1)[:, 0]
        return evaluate_forecast(y_true, y_pred, previous)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        (directory / "metadata.json").write_text(
            json.dumps({"feature_names": self.feature_names, "metadata": self.metadata}, indent=2),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "LinearRegressionForecastModel":
        instance = cls()
        instance.model = joblib.load(directory / "model.joblib")
        instance.scaler = joblib.load(directory / "scaler.joblib")
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance.feature_names = payload.get("feature_names", [])
        instance.metadata = payload.get("metadata", {})
        instance.is_fitted = True
        return instance
