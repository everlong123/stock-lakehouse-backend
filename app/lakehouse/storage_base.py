"""Abstract object/file storage used by the lakehouse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO

import pandas as pd


class StorageBackend(ABC):
    """Read/write Parquet datasets for Bronze, Silver, and Gold."""

    backend_name: str = "abstract"

    @abstractmethod
    def write_parquet(self, layer: str, relative_path: str, frame: pd.DataFrame) -> str:
        """Persist a DataFrame as Parquet and return the storage path."""

    @abstractmethod
    def read_parquet(self, layer: str, relative_path: str) -> pd.DataFrame:
        """Read a single Parquet object."""

    @abstractmethod
    def read_prefix(self, layer: str, prefix: str) -> pd.DataFrame:
        """Read and concatenate all Parquet files under a prefix."""

    @abstractmethod
    def list_objects(self, layer: str, prefix: str = "") -> list[str]:
        """List object keys under a prefix."""

    @abstractmethod
    def exists(self, layer: str, relative_path: str) -> bool:
        """Return True if the object exists."""

    @abstractmethod
    def write_json(self, layer: str, relative_path: str, payload: dict) -> str:
        """Persist a JSON metadata document."""

    @abstractmethod
    def read_json(self, layer: str, relative_path: str) -> dict:
        """Read a JSON metadata document."""

    @abstractmethod
    def count_objects(self, layer: str, prefix: str = "") -> int:
        """Count objects under a prefix."""

    def health(self) -> dict[str, str | bool]:
        """Return a status payload for the System Status page."""
        return {"backend": self.backend_name, "available": True}


def dataframe_to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes."""
    buffer = BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def parquet_bytes_to_dataframe(payload: bytes) -> pd.DataFrame:
    """Deserialize Parquet bytes into a DataFrame."""
    return pd.read_parquet(BytesIO(payload), engine="pyarrow")
