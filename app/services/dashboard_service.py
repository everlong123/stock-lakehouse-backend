"""Dashboard aggregation service."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import get_logger
from app.database.repositories.backtest_repository import BacktestRepository
from app.database.repositories.model_repository import ModelRepository
from app.database.repositories.pipeline_repository import PipelineRepository
from app.database.session import check_database_connection
from app.forecasting.model_registry import list_trained_models
from app.indicators.service import describe_indicators
from app.lakehouse.bronze import BronzeLayer
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer
from app.services.market_service import MarketService

logger = get_logger(__name__)


class DashboardService:
    def __init__(self) -> None:
        self.market = MarketService()

    def build(self, symbol: str, db: Session | None = None) -> dict:
        history = self.market.get_history(symbol)
        latest = self.market.get_latest(symbol)
        last = history.iloc[-1]
        rsi_value = None
        if "rsi_14" in history.columns and pd.notna(last["rsi_14"]):
            rsi_value = float(last["rsi_14"])
        summary = describe_indicators(last) if "sma_20" in history.columns else {}
        candles = self.market.to_records(history, limit=250)
        prediction = None
        models = list_trained_models(symbol)
        if models:
            best = models[0]
            prediction = {
                "model_name": best.get("model_name"),
                "mae": best.get("mae"),
                "rmse": best.get("rmse"),
                "mape": best.get("mape"),
                "directional_accuracy": best.get("directional_accuracy"),
            }
        pipeline = None
        backtest = None
        if db is not None and check_database_connection():
            try:
                latest_pipeline = PipelineRepository(db).latest(symbol)
                if latest_pipeline:
                    pipeline = {
                        "status": latest_pipeline.status,
                        "records_processed": latest_pipeline.records_processed,
                        "error_count": latest_pipeline.error_count,
                        "message": latest_pipeline.message,
                        "created_at": latest_pipeline.created_at.isoformat() if latest_pipeline.created_at else None,
                    }
                latest_model = ModelRepository(db).latest(symbol)
                if latest_model:
                    prediction = {
                        "model_name": latest_model.model_name,
                        "mae": latest_model.mae,
                        "rmse": latest_model.rmse,
                        "mape": latest_model.mape,
                        "directional_accuracy": latest_model.directional_accuracy,
                    }
                latest_bt = BacktestRepository(db).latest(symbol)
                if latest_bt:
                    backtest = {
                        "strategy": latest_bt.strategy,
                        "total_return": latest_bt.total_return,
                        "sharpe_ratio": latest_bt.sharpe_ratio,
                        "maximum_drawdown": latest_bt.maximum_drawdown,
                        "win_rate": latest_bt.win_rate,
                    }
            except Exception as exc:
                logger.warning("Dashboard metadata lookup failed: %s", exc)
        return {
            "symbol": symbol.upper(),
            "current_price": latest["close"],
            "change_pct": latest["change_pct"],
            "volume": latest["volume"],
            "rsi": rsi_value,
            "latest_prediction": prediction,
            "latest_backtest": backtest,
            "pipeline": pipeline,
            "summary": summary,
            "candles": candles,
            "counts": {
                "bronze": BronzeLayer().record_count(symbol),
                "silver": SilverLayer().record_count(symbol),
                "gold": GoldLayer().record_count(symbol),
            },
            "data_source": settings.data_source,
            "disclaimer": (
                "This dashboard is a research prototype. Forecasts and backtests are not investment advice."
            ),
        }
