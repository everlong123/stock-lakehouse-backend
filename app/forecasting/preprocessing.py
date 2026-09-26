"""Chronological train/validation/test split. Never shuffle time series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core.config import settings
from app.core.exceptions import ModelTrainingError


@dataclass
class TimeSeriesSplit:
    """Ordered partitions of a feature frame."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float | None = None,
    validation_ratio: float | None = None,
    test_ratio: float | None = None,
) -> TimeSeriesSplit:
    """Split a time-sorted frame into train/validation/test without shuffling."""
    train_ratio = train_ratio if train_ratio is not None else settings.train_ratio
    validation_ratio = validation_ratio if validation_ratio is not None else settings.validation_ratio
    test_ratio = test_ratio if test_ratio is not None else settings.test_ratio
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ModelTrainingError("Train/validation/test ratios must sum to 1.")
    if "timestamp" in frame.columns:
        ordered = frame.sort_values("timestamp").reset_index(drop=True)
    else:
        ordered = frame.reset_index(drop=True)
    n = len(ordered)
    if n < 30:
        raise ModelTrainingError("Need at least 30 rows after feature warmup to train models.")
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * validation_ratio)
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()
    if train.empty or validation.empty or test.empty:
        raise ModelTrainingError("One of the time-series splits is empty. Provide more history.")
    return TimeSeriesSplit(train=train, validation=validation, test=test)
