"""Authentication primitives: password hashing, signed tokens, roles, login throttling."""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt

logger = logging.getLogger(__name__)
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MIN_SECRET_LENGTH = 32
MIN_PASSWORD_LENGTH = 12


class Role(StrEnum):
    ANALYST = "analyst"  # views and decides cases
    AUDITOR = "auditor"  # reads cases (no personal details) and the audit trail; cannot decide
    ADMIN = "admin"  # manages users; cannot decide cases (separation of duties)


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64), dklen=len(expected), **_SCRYPT
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def resolve_secret(configured: str, app_env: str) -> str:
    """A strong secret is mandatory outside development; in development a weak or default
    value is replaced by a random per-process secret (tokens then do not survive a restart)."""
    if len(configured) >= MIN_SECRET_LENGTH and configured != "change-me":
        return configured
    if app_env == "development":
        logger.warning(
            "API_SECRET_KEY is weak or unset; using a random per-process secret (dev only)"
        )
        return secrets.token_urlsafe(48)
    raise RuntimeError(f"API_SECRET_KEY must be set to at least {MIN_SECRET_LENGTH} characters")


def create_token(username: str, role: Role, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": username,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict[str, Any]:
    """Raises jwt.PyJWTError for any invalid, expired or tampered token; only HS256 is accepted."""
    claims: dict[str, Any] = jwt.decode(
        token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "role"]}
    )
    return claims


class LoginLimiter:
    """In-memory throttle: after ``max_failures`` failures within ``window_s`` a key is locked out.

    Process-local (one API instance). A multi-instance deployment needs a shared store.
    """

    def __init__(self, max_failures: int = 5, window_s: float = 300.0) -> None:
        self.max_failures, self.window_s = max_failures, window_s
        self._failures: dict[str, list[float]] = {}

    def _recent(self, key: str) -> list[float]:
        cutoff = time.monotonic() - self.window_s
        recent = [t for t in self._failures.get(key, []) if t > cutoff]
        self._failures[key] = recent
        return recent

    def is_locked(self, key: str) -> bool:
        return len(self._recent(key)) >= self.max_failures

    def record_failure(self, key: str) -> None:
        self._recent(key).append(time.monotonic())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
