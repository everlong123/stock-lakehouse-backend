"""Shared API envelope and helpers."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard success/error envelope used by every endpoint."""

    success: bool = True
    data: T | None = None
    message: str | None = None


class Pagination(BaseModel):
    limit: int = Field(default=100, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)


def ok(data: Any, message: str | None = None) -> dict[str, Any]:
    return {"success": True, "data": data, "message": message}


def fail(message: str) -> dict[str, Any]:
    return {"success": False, "data": None, "message": message}
