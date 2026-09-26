"""Domain-specific exceptions used across the platform."""

from __future__ import annotations


class StockLakehouseError(Exception):
    """Base exception for the platform."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DataSourceError(StockLakehouseError):
    """Raised when a market data provider fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class DataValidationError(StockLakehouseError):
    """Raised when incoming or transformed data fails validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class StorageError(StockLakehouseError):
    """Raised when lakehouse storage operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class DatabaseError(StockLakehouseError):
    """Raised when MySQL operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503)


class ModelTrainingError(StockLakehouseError):
    """Raised when model training cannot complete."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class PredictionError(StockLakehouseError):
    """Raised when a forecast cannot be produced."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class BacktestError(StockLakehouseError):
    """Raised when a backtest cannot run."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class AgentError(StockLakehouseError):
    """Raised when the AI agent cannot complete a request."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class NotFoundError(StockLakehouseError):
    """Raised when a requested resource is missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)
