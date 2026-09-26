"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, previous: np.ndarray) -> float:
    """Share of periods where predicted direction from previous close matches actual."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)
    actual_dir = np.sign(y_true - previous)
    pred_dir = np.sign(y_pred - previous)
    mask = actual_dir != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(actual_dir[mask] == pred_dir[mask]))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray | None = None,
) -> dict[str, float]:
    """Return the standard academic metric set. No claim that any model is best."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "directional_accuracy": 0.0,
    }
    if previous_close is not None and len(previous_close) == len(y_true):
        metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred, previous_close)
    return metrics
