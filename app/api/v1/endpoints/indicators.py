"""Technical indicator endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_indicator_service
from app.schemas.common import ok
from app.services.indicator_service import IndicatorService

router = APIRouter(prefix="/stocks", tags=["indicators"])


@router.get("/{symbol}/indicators")
def get_indicators(
    symbol: str,
    sma: bool = True,
    ema: bool = True,
    rsi: bool = True,
    macd: bool = True,
    bollinger: bool = True,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    service: IndicatorService = Depends(get_indicator_service),
) -> dict:
    data = service.calculate(
        symbol.upper(),
        sma=sma,
        ema=ema,
        rsi=rsi,
        macd=macd,
        bollinger=bollinger,
        rsi_period=rsi_period,
        bb_window=bb_window,
        bb_std=bb_std,
    )
    return ok(data)
