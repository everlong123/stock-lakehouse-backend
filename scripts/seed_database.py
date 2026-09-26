"""Seed the admin user used by the research prototype."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.core.logging_config import get_logger
from app.core.security import hash_password
from app.database.models.user import User
from app.database.session import get_session_factory, require_database

logger = get_logger(__name__)


def main() -> None:
    require_database()
    session = get_session_factory()()
    try:
        existing = session.scalar(select(User).where(User.username == "admin"))
        if existing is None:
            session.add(
                User(
                    username="admin",
                    email="admin@stock-lakehouse.local",
                    password_hash=hash_password("ChangeMe123!"),
                    role="admin",
                )
            )
            session.commit()
            logger.info("Seeded admin user (username=admin). Change the password before any shared demo.")
        else:
            logger.info("Admin user already exists.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
