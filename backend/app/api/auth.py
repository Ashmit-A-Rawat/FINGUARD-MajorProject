from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import CurrentUser, current_user, get_session, require_roles
from backend.app.core.security import Role, create_token, hash_password, verify_password
from backend.app.database.models import UserRow
from backend.app.schemas.api import CreateUserRequest, LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/api", tags=["auth"])
_DUMMY_HASH = hash_password("dummy-password-for-timing")  # equalises timing for unknown users


@router.post("/auth/login", response_model=TokenResponse)
def login(
    body: LoginRequest, request: Request, session: Annotated[Session, Depends(get_session)]
) -> TokenResponse:
    limiter = request.app.state.limiter
    key = f"{request.client.host if request.client else 'unknown'}:{body.username.lower()}"
    if limiter.is_locked(key):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "too many failed attempts; try later"
        )
    user = session.execute(
        select(UserRow).where(UserRow.username == body.username.lower())
    ).scalar_one_or_none()
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if not (ok and user and user.active):
        limiter.record_failure(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    limiter.reset(key)
    settings = request.app.state.settings
    return TokenResponse(
        access_token=create_token(
            user.username, Role(user.role), request.app.state.secret, settings.token_ttl_minutes
        ),
        username=user.username,
        role=Role(user.role),
        expires_in_minutes=settings.token_ttl_minutes,
    )


@router.get("/auth/me", response_model=UserOut)
def me(user: Annotated[CurrentUser, Depends(current_user)]) -> UserOut:
    return UserOut(username=user.username, role=user.role)


@router.post("/admin/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    _: Annotated[CurrentUser, Depends(require_roles(Role.ADMIN))],
    session: Annotated[Session, Depends(get_session)],
) -> UserOut:
    if session.execute(select(UserRow).where(UserRow.username == body.username)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "username already exists")
    session.add(
        UserRow(
            username=body.username,
            password_hash=hash_password(body.password),
            role=body.role.value,
            active=True,
            created_at=datetime.utcnow(),
        )
    )
    return UserOut(username=body.username, role=body.role)
