"""Forecasting endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_forecasting_service
from app.schemas.common import ok
from app.schemas.forecasting import PredictRequest, TrainRequest
from app.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["forecasting"])


@router.post("/train")
def train_forecast(
    payload: TrainRequest,
    db: Session | None = Depends(get_db),
    service: ForecastingService = Depends(get_forecasting_service),
) -> dict:
    result = service.train(payload.model_dump(), db=db)
    return ok(result, message="Model trained on a chronological split. Metrics are not a profit guarantee.")


@router.post("/predict")
def predict_forecast(
    payload: PredictRequest,
    service: ForecastingService = Depends(get_forecasting_service),
) -> dict:
    return ok(service.predict(payload.symbol.upper(), payload.model_name, payload.horizon))


@router.get("/compare/{symbol}")
def compare_forecast(
    symbol: str,
    db: Session | None = Depends(get_db),
    service: ForecastingService = Depends(get_forecasting_service),
) -> dict:
    return ok(service.compare(symbol.upper(), db=db))
