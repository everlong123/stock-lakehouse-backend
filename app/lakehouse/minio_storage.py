"""MinIO object storage backend."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.logging_config import get_logger
from app.lakehouse.storage_base import (
    StorageBackend,
    dataframe_to_parquet_bytes,
    parquet_bytes_to_dataframe,
)

logger = get_logger(__name__)

LAYER_BUCKETS = {
    "bronze": settings.minio_bucket_bronze,
    "silver": settings.minio_bucket_silver,
    "gold": settings.minio_bucket_gold,
    "models": settings.minio_bucket_models,
    "backtests": settings.minio_bucket_backtests,
    "quality": settings.minio_bucket_silver,
}


class MinioStorageBackend(StorageBackend):
    """Store lakehouse datasets in MinIO buckets."""

    backend_name = "minio"

    def __init__(self) -> None:
        try:
            from minio import Minio
        except ImportError as exc:
            raise StorageError("minio package is not installed.") from exc
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._ensure_buckets()

    def _ensure_buckets(self) -> None:
        try:
            for bucket in set(LAYER_BUCKETS.values()):
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info("Created MinIO bucket %s", bucket)
        except Exception as exc:
            raise StorageError(f"Cannot initialize MinIO buckets: {exc}") from exc

    def _bucket(self, layer: str) -> str:
        if layer not in LAYER_BUCKETS:
            raise StorageError(f"Unknown lakehouse layer: {layer}")
        return LAYER_BUCKETS[layer]

    def write_parquet(self, layer: str, relative_path: str, frame: pd.DataFrame) -> str:
        payload = dataframe_to_parquet_bytes(frame)
        return self._put_bytes(layer, relative_path, payload, "application/octet-stream")

    def read_parquet(self, layer: str, relative_path: str) -> pd.DataFrame:
        payload = self._get_bytes(layer, relative_path)
        return parquet_bytes_to_dataframe(payload)

    def read_prefix(self, layer: str, prefix: str) -> pd.DataFrame:
        keys = [key for key in self.list_objects(layer, prefix) if key.endswith(".parquet")]
        if not keys:
            return pd.DataFrame()
        frames = [self.read_parquet(layer, key) for key in keys]
        return pd.concat(frames, ignore_index=True)

    def list_objects(self, layer: str, prefix: str = "") -> list[str]:
        bucket = self._bucket(layer)
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except Exception as exc:
            raise StorageError(f"MinIO list failed: {exc}") from exc

    def exists(self, layer: str, relative_path: str) -> bool:
        bucket = self._bucket(layer)
        try:
            self.client.stat_object(bucket, relative_path)
            return True
        except Exception:
            return False

    def write_json(self, layer: str, relative_path: str, payload: dict) -> str:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        return self._put_bytes(layer, relative_path, body, "application/json")

    def read_json(self, layer: str, relative_path: str) -> dict:
        payload = self._get_bytes(layer, relative_path)
        return json.loads(payload.decode("utf-8"))

    def count_objects(self, layer: str, prefix: str = "") -> int:
        return len(self.list_objects(layer, prefix))

    def health(self) -> dict[str, str | bool]:
        try:
            self.client.list_buckets()
            return {"backend": self.backend_name, "available": True}
        except Exception as exc:
            logger.warning("MinIO health check failed: %s", exc)
            return {"backend": self.backend_name, "available": False}

    def _put_bytes(self, layer: str, relative_path: str, payload: bytes, content_type: str) -> str:
        bucket = self._bucket(layer)
        try:
            self.client.put_object(
                bucket,
                relative_path,
                BytesIO(payload),
                length=len(payload),
                content_type=content_type,
            )
        except Exception as exc:
            raise StorageError(f"MinIO put failed for {relative_path}: {exc}") from exc
        return f"{bucket}/{relative_path}"

    def _get_bytes(self, layer: str, relative_path: str) -> bytes:
        bucket = self._bucket(layer)
        try:
            response = self.client.get_object(bucket, relative_path)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            raise StorageError(f"MinIO get failed for {relative_path}: {exc}") from exc
