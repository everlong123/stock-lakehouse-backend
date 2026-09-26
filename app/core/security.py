"""Password hashing helpers. Never store plaintext passwords."""

from __future__ import annotations

from passlib.context import CryptContext


_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a bcrypt hash for the given password."""
    if not password:
        raise ValueError("Password must not be empty.")
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    if not password or not password_hash:
        return False
    return _pwd_context.verify(password, password_hash)
