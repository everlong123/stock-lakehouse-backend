"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import StockLakehouseError
from app.core.logging_config import get_logger, setup_logging
from app.core.seeding import set_global_seed
from app.schemas.common import fail
from app.services.crawler_service import start_crawler, stop_crawler

setup_logging()
set_global_seed(settings.random_seed)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Nền tảng Data Lakehouse phân tích và dự báo chứng khoán phục vụ nghiên cứu học thuật. "
        "Kết quả không phải khuyến nghị đầu tư và không cam kết lợi nhuận."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.exception_handler(StockLakehouseError)
async def domain_exception_handler(_request: Request, exc: StockLakehouseError) -> JSONResponse:
    logger.warning("Domain error: %s", exc.message)
    return JSONResponse(status_code=exc.status_code, content=fail(exc.message))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content=fail("Internal server error."))


@app.get("/")
def root() -> dict:
    return {
        "success": True,
        "data": {
            "name": settings.app_name,
            "docs": "/docs",
            "api": settings.api_prefix,
        },
        "message": "Research prototype. Not investment advice.",
    }


@app.on_event("startup")
async def startup_event():
    """Start the stock crawler on app startup."""
    crawler = start_crawler()
    if crawler:
        logger.info("Stock crawler started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the stock crawler on app shutdown."""
    stop_crawler()
    logger.info("Stock crawler stopped")
