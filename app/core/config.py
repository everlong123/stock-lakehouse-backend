"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration for the Stock Lakehouse platform."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    app_name: str = "Stock Lakehouse AI"
    app_env: Literal["development", "production", "test"] = "development"
    app_debug: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175"

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "stock_lakehouse"
    mysql_user: str = "root"
    mysql_password: str = "123456"
    database_url: str = "mysql+pymysql://root:123456@localhost:3306/stock_lakehouse"

    storage_backend: Literal["local", "minio"] = "local"
    data_root: str = "data"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_bronze: str = "stock-bronze"
    minio_bucket_silver: str = "stock-silver"
    minio_bucket_gold: str = "stock-gold"
    minio_bucket_models: str = "stock-models"
    minio_bucket_backtests: str = "stock-backtests"

    use_spark: bool = False
    spark_app_name: str = "stock-lakehouse"
    spark_master: str = "local[*]"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: int = 60

    default_symbol: str = "AAPL"
    default_interval: str = "1d"
    default_lookback_days: int = 730

    model_dir: str = "data/models"
    random_seed: int = 42

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 32
    lstm_epochs: int = 20
    lstm_patience: int = 5

    arima_order: str = "5,1,0"

    data_source: Literal["sample", "yfinance", "alpha_vantage", "web_scraper"] = "sample"
    yfinance_timeout: int = 30
    alpha_vantage_api_key: str = ""

    # Crawler settings
    crawl_enabled: bool = True
    crawl_interval_seconds: int = 300  # 5 minutes
    # 25 stocks across 9 sectors: Banking, Securities, Real Estate, Tech, Retail, Steel, Oil & Gas, Power, FMCG
    crawl_symbols: str = "VCB,TCB,MBB,ACB,BID,SSI,VND,VHM,VRE,KDH,FPT,CMG,MWG,HPG,GAS,PLX,POW,VNM,SAB,MSN,VIC,VPB,CTG,TPB,SHB,STB,PNJ,HDB,LPB,MSB,OCB,REE,NVL,PDR,NVL,BCM,SBT,IMP,KDC,PC1,HDG,DRC,DXG,IDJ,ITA,JVC,LSG,MSH, NSC,PVT,mbc,dig,fcn,hcm,ctc,fpt,smt,ksc,vgc,bvh,c22,c32"

    # ── Apache Iceberg ────────────────────────────────────────────────────────
    use_iceberg: bool = False  # Set True to use Iceberg instead of raw Parquet
    iceberg_catalog_uri: str = "http://localhost:8181"
    iceberg_warehouse: str = "s3://stock-lakehouse/iceberg"

    # ── Apache Kafka (Real-time Streaming) ────────────────────────────────────
    use_kafka: bool = False  # Set True to enable Kafka streaming
    kafka_bootstrap_servers: str = "localhost:9094"
    kafka_topic_raw: str = "stock-ohlcv-raw"
    kafka_consumer_group: str = "stock-lakehouse-consumer"

    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    @field_validator("train_ratio", "validation_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            raise ValueError("Split ratios must be between 0 and 1.")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def backend_root(self) -> Path:
        return BACKEND_ROOT

    @property
    def data_root_path(self) -> Path:
        path = Path(self.data_root)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def model_dir_path(self) -> Path:
        path = Path(self.model_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def log_file_path(self) -> Path:
        path = Path(self.log_file)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def arima_order_tuple(self) -> tuple[int, int, int]:
        parts = [int(item.strip()) for item in self.arima_order.split(",")]
        if len(parts) != 3:
            raise ValueError("ARIMA_ORDER must be three integers, e.g. 5,1,0")
        return parts[0], parts[1], parts[2]

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
