"""Strategy interface. Signals are generated from bar t and executed at t+1."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Research strategy that emits 1 (buy), -1 (sell), or 0 (hold)."""

    name: str = "base"

    def __init__(self, parameters: dict | None = None) -> None:
        self.parameters = parameters or {}

    @abstractmethod
    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        """Return a signal series aligned with `frame`. Must not use future bars."""
