"""Health and system status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging_config import get_logger
from app.database.session import check_database_connection
from app.lakehouse.bronze import BronzeLayer
from app.lakehouse.gold import GoldLayer
from app.lakehouse.silver import SilverLayer
from app.lakehouse.spark_session import spark_enabled
from app.lakehouse.storage_factory import get_storage_backend
from app.schemas.common import ok

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return ok({"status": "ok", "app": settings.app_name})


@router.get("/system/status")
def system_status() -> dict:
    storage = get_storage_backend()
    mysql_ok = check_database_connection()
    minio_mode = storage.backend_name == "minio"
    agent_ready = bool(settings.openai_api_key.strip())
    payload = {
        "backend_api": {"label": "Backend API", "status": "online"},
        "mysql": {"label": "MySQL", "status": "connected" if mysql_ok else "disconnected"},
        "minio": {
            "label": "MinIO",
            "status": "connected" if minio_mode else "local_storage_mode",
        },
        "spark": {
            "label": "Spark",
            "status": "enabled" if spark_enabled() else "disabled_pandas_mode",
        },
        "data_source": {"label": "Data Source", "status": settings.data_source},
        "ai_agent": {
            "label": "AI Agent",
            "status": "llm_ready" if agent_ready else "local_tool_router",
        },
        "counts": {
            "bronze": BronzeLayer().record_count(),
            "silver": SilverLayer().record_count(),
            "gold": GoldLayer().record_count(),
        },
        "storage_backend": storage.backend_name,
        "use_spark": settings.use_spark,
    }
    return ok(payload)
