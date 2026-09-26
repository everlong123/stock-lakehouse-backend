"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.database.session import check_database_connection, get_session_factory
from app.services.agent_service import AgentService
from app.services.backtest_service import BacktestService
from app.services.dashboard_service import DashboardService
from app.services.forecasting_service import ForecastingService
from app.services.indicator_service import IndicatorService
from app.services.market_service import MarketService
from app.services.pipeline_service import PipelineService

logger = get_logger(__name__)


def get_db() -> Generator[Session | None, None, None]:
    """Yield a MySQL session when available, otherwise None."""
    if not check_database_connection():
        yield None
        return
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_market_service() -> MarketService:
    return MarketService()


def get_indicator_service() -> IndicatorService:
    return IndicatorService()


def get_forecasting_service() -> ForecastingService:
    return ForecastingService()


def get_backtest_service() -> BacktestService:
    return BacktestService()


def get_dashboard_service() -> DashboardService:
    return DashboardService()


def get_pipeline_service() -> PipelineService:
    return PipelineService()


def get_agent_service() -> AgentService:
    return AgentService()
