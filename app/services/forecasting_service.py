"""Forecasting application service."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.database.repositories.model_repository import ModelRepository
from app.database.session import check_database_connection
from app.forecasting.model_registry import list_trained_models
from app.forecasting.predictor import predict_symbol
from app.forecasting.trainer import train_model

logger = get_logger(__name__)


class ForecastingService:
    def train(self, payload: dict[str, Any], db: Session | None = None) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if payload.get("model_name") == "lstm":
            extra = {
                "sequence_length": payload.get("sequence_length"),
                "hidden_size": payload.get("hidden_size"),
                "num_layers": payload.get("num_layers"),
                "epochs": payload.get("epochs"),
            }
            extra = {k: v for k, v in extra.items() if v is not None}
        if payload.get("model_name") == "arima":
            extra["order"] = (
                payload.get("arima_p") or 5,
                payload.get("arima_d") or 1,
                payload.get("arima_q") or 0,
            )
        result = train_model(payload["symbol"], payload["model_name"], payload.get("horizon", 1), extra)
        self._persist(result, db)
        return result

    def predict(self, symbol: str, model_name: str, horizon: int = 5) -> dict[str, Any]:
        return predict_symbol(symbol, model_name, horizon)

    def compare(self, symbol: str, db: Session | None = None) -> dict[str, Any]:
        records = list_trained_models(symbol)
        if db is not None and check_database_connection():
            try:
                db_rows = ModelRepository(db).comparison_rows(symbol)
                if db_rows:
                    records = [
                        {
                            "symbol": row.symbol,
                            "model_name": row.model_name,
                            "mae": row.mae,
                            "rmse": row.rmse,
                            "mape": row.mape,
                            "directional_accuracy": row.directional_accuracy,
                            "created_at": row.created_at.isoformat() if row.created_at else None,
                            "parameters": row.parameters,
                        }
                        for row in db_rows
                    ]
            except Exception as exc:
                logger.warning("Could not read model_runs from MySQL: %s", exc)
        return {
            "symbol": symbol.upper(),
            "models": records,
            "note": "Comparison uses hold-out metrics. A lower error does not guarantee a better live forecast.",
        }

    def _persist(self, result: dict[str, Any], db: Session | None) -> None:
        if db is None or not check_database_connection():
            return
        try:
            ModelRepository(db).create(
                {
                    "symbol": result["symbol"],
                    "model_name": result["model_name"],
                    "train_start": _as_dt(result.get("train_start")),
                    "train_end": _as_dt(result.get("train_end")),
                    "validation_start": _as_dt(result.get("validation_start")),
                    "validation_end": _as_dt(result.get("validation_end")),
                    "test_start": _as_dt(result.get("test_start")),
                    "test_end": _as_dt(result.get("test_end")),
                    "features": ",".join(result.get("features") or []),
                    "parameters": str(result.get("parameters")),
                    "mae": result.get("mae"),
                    "rmse": result.get("rmse"),
                    "mape": result.get("mape"),
                    "directional_accuracy": result.get("directional_accuracy"),
                    "model_path": result.get("model_path"),
                }
            )
        except Exception as exc:
            logger.warning("Unable to persist model run to MySQL: %s", exc)


def _as_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return pd_to_datetime(value)


def pd_to_datetime(value) -> datetime | None:
    import pandas as pd

    ts = pd.to_datetime(value, utc=True)
    return ts.to_pydatetime().replace(tzinfo=None)
