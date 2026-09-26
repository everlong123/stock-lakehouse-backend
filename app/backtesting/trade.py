"""A single closed trade. Execution always occurs on a later bar than the signal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """One completed round-trip position."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    return_pct: float
    fees: float
