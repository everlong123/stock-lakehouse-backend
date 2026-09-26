"""Storage backend factory with local fallback when MinIO is unavailable."""

from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import get_logger
from app.lakehouse.local_storage import LocalStorageBackend
from app.lakehouse.storage_base import StorageBackend

logger = get_logger(__name__)


def get_storage_backend() -> StorageBackend:
    """Return MinIO storage when configured and healthy, otherwise local files."""
    if settings.storage_backend == "minio":
        try:
            from app.lakehouse.minio_storage import MinioStorageBackend

            backend = MinioStorageBackend()
            health = backend.health()
            if health.get("available"):
                logger.info("Using MinIO storage backend.")
                return backend
            logger.warning("MinIO is configured but unavailable. Falling back to local storage.")
        except Exception as exc:
            logger.warning("MinIO initialization failed (%s). Falling back to local storage.", exc)
    logger.info("Using local filesystem storage backend.")
    return LocalStorageBackend()
