"""Request dependencies: database session, authenticated user, role checks."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.security import Role, decode_token
from backend.app.database.models import UserRow

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    username: str
    role: Role


def get_session(request: Request) -> Iterator[Session]:
    """One transaction per request: commit on success, roll back on any error."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Annotated[Session, Depends(get_session)],
) -> CurrentUser:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    try:
        claims = decode_token(credentials.credentials, request.app.state.secret)
    except jwt.PyJWTError:
        raise unauthorized from None
    # The database, not the token, is the source of truth for role and active status.
    user = session.execute(
        select(UserRow).where(UserRow.username == claims["sub"])
    ).scalar_one_or_none()
    if user is None or not user.active:
        raise unauthorized
    return CurrentUser(user.username, Role(user.role))


def require_roles(*allowed: Role) -> Callable[[CurrentUser], CurrentUser]:
    def check(user: Annotated[CurrentUser, Depends(current_user)]) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{user.role}' is not permitted")
        return user

    return check
