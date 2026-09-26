"""Backtesting endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_backtest_service, get_db
from app.schemas.backtesting import BacktestRequest
from app.schemas.common import ok
from app.services.backtest_service import BacktestService

router = APIRouter(prefix="/backtests", tags=["backtesting"])


@router.post("/run")
def run_backtest(
    payload: BacktestRequest,
    db: Session | None = Depends(get_db),
    service: BacktestService = Depends(get_backtest_service),
) -> dict:
    result = service.run(payload.model_dump(), db=db)
    return ok(result, message="Historical backtest finished. This is not live trading.")


@router.get("/history")
def backtest_history(
    symbol: str | None = Query(default=None),
    db: Session | None = Depends(get_db),
    service: BacktestService = Depends(get_backtest_service),
) -> dict:
    return ok(service.history(symbol, db=db))
