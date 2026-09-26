"""Create the MySQL database and apply migrations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.logging_config import get_logger
from app.database.base import Base
from app.database.models import AgentConversation, BacktestRun, ModelRun, PipelineRun, User  # noqa: F401
from app.database.session import get_engine

logger = get_logger(__name__)


def create_database_if_needed() -> None:
    admin_url = (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/?charset=utf8mb4"
    )
    engine = create_engine(admin_url, future=True)
    with engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
        connection.commit()
    engine.dispose()
    logger.info("Database %s is ready.", settings.mysql_database)


def run_migrations() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", settings.sqlalchemy_url)
    try:
        command.upgrade(cfg, "head")
        logger.info("Alembic upgrade head complete.")
    except Exception as exc:
        logger.warning("Alembic upgrade failed (%s). Falling back to metadata.create_all.", exc)
        Base.metadata.create_all(bind=get_engine())
        command.stamp(cfg, "head")
        logger.info("Tables created and Alembic stamped at head.")


def main() -> None:
    try:
        create_database_if_needed()
        run_migrations()
        logger.info("MySQL initialization complete.")
    except Exception as exc:
        logger.error(
            "MySQL initialization failed: %s. "
            "Create the database manually:\n"
            "CREATE DATABASE stock_lakehouse CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
            exc,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
