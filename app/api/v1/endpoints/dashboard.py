"""Dashboard endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_dashboard_service, get_db
from app.schemas.common import ok
from app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/{symbol}")
def get_dashboard(
    symbol: str,
    db: Session | None = Depends(get_db),
    service: DashboardService = Depends(get_dashboard_service),
) -> dict:
    return ok(service.build(symbol.upper(), db=db))
