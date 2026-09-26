"""Dashboard payload fragments."""

from __future__ import annotations

from pydantic import BaseModel


class DashboardQuery(BaseModel):
    interval: str = "1d"
