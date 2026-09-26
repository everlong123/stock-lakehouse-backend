"""Aggregate v1 routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import agent, backtesting, dashboard, data, forecasting, health, indicators, market, pipeline, stocks

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(dashboard.router)
api_router.include_router(stocks.router)
api_router.include_router(indicators.router)
api_router.include_router(pipeline.router)
api_router.include_router(forecasting.router)
api_router.include_router(backtesting.router)
api_router.include_router(agent.router)
api_router.include_router(data.router)
api_router.include_router(market.router)
