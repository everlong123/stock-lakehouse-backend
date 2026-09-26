"""Shared forecasting contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseForecastModel(ABC):
    """All forecast models implement fit/predict/evaluate/save/load."""

    model_name: str = "base"

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseForecastModel":
        """Train the model on chronological training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return next-close predictions aligned with X."""

    @abstractmethod
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
        """Return MAE, RMSE, MAPE, and directional accuracy."""

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Persist model artifacts into directory."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseForecastModel":
        """Load a previously saved model."""
