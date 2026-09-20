import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from api_helpers import PASSWORD, SECRET, Api

from backend.app.core.config import Settings
from backend.app.core.security import (
    LoginLimiter,
    Role,
    hash_password,
    resolve_secret,
    verify_password,
)
from backend.app.main import create_app


def test_passwords_are_hashed_salted_and_verified_in_constant_form() -> None:
    a, b = hash_password(PASSWORD), hash_password(PASSWORD)
    assert a != b and PASSWORD not in a and a.startswith("scrypt$")
    assert verify_password(PASSWORD, a) and not verify_password("wrong password!!", a)
    assert not verify_password(PASSWORD, "garbage") and not verify_password(PASSWORD, "md5$a$b")
    with pytest.raises(ValueError):
        hash_password("short")


def test_secret_resolution() -> None:
    assert resolve_secret(SECRET, "production") == SECRET
    with pytest.raises(RuntimeError):
        resolve_secret("change-me", "production")
    with pytest.raises(RuntimeError):
        resolve_secret("too-short", "staging")
    dev = resolve_secret("change-me", "development")
    assert len(dev) >= 32 and dev != "change-me"


def test_app_refuses_to_start_with_a_weak_secret_outside_development() -> None:
    with pytest.raises(RuntimeError):
        create_app(Settings(app_env="production", api_secret_key="weak", engine_enabled=False))


def test_login_success_and_failure_do_not_reveal_which_part_was_wrong(api: Api) -> None:
    ok = api.client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert ok.status_code == 200 and ok.json()["role"] == "analyst"
    wrong_pw = api.client.post(
        "/api/auth/login", json={"username": "alice", "password": "nope nope nope"}
    )
    unknown = api.client.post("/api/auth/login", json={"username": "nobody", "password": PASSWORD})
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json() == unknown.json()


def test_repeated_failures_lock_the_account_for_that_client(api: Api) -> None:
    for _ in range(5):
        assert (
            api.client.post(
                "/api/auth/login", json={"username": "alice", "password": "bad bad bad bad"}
            ).status_code
            == 401
        )
    assert (
        api.client.post(
            "/api/auth/login", json={"username": "alice", "password": PASSWORD}
        ).status_code
        == 429
    )


def test_limiter_window() -> None:
    limiter = LoginLimiter(max_failures=2, window_s=0.2)
    limiter.record_failure("k")
    limiter.record_failure("k")
    assert limiter.is_locked("k")
    time.sleep(0.25)
    assert not limiter.is_locked("k")


def test_inactive_user_cannot_use_an_existing_token_or_log_in(api: Api) -> None:
    api.add_user("carol", Role.ANALYST)
    headers = api.headers("carol")
    assert api.client.get("/api/auth/me", headers=headers).status_code == 200
    from sqlalchemy import update

    from backend.app.database.models import UserRow

    with api.app.state.session_factory() as s:  # type: ignore[attr-defined]
        s.execute(update(UserRow).where(UserRow.username == "carol").values(active=False))
        s.commit()
    assert api.client.get("/api/auth/me", headers=headers).status_code == 401
    assert (
        api.client.post(
            "/api/auth/login", json={"username": "carol", "password": PASSWORD}
        ).status_code
        == 401
    )


def test_role_comes_from_the_database_not_the_token(api: Api) -> None:
    headers = api.headers("audrey")
    forged = jwt.encode(
        {"sub": "audrey", "role": "analyst", "exp": datetime.now(UTC) + timedelta(hours=1)},
        SECRET,
        algorithm="HS256",
    )
    r = api.client.post(
        "/api/cases",
        json={"customer_id": "CUST-000001"},
        headers={"Authorization": f"Bearer {forged}"},
    )
    assert r.status_code == 403  # the auditor's real role wins
    assert api.client.get("/api/auth/me", headers=headers).json()["role"] == "auditor"


@pytest.mark.parametrize("bad", ["", "Bearer", "Bearer not.a.token", "Basic abc"])
def test_missing_or_malformed_credentials_are_rejected(api: Api, bad: str) -> None:
    headers = {"Authorization": bad} if bad else {}
    assert api.client.get("/api/cases", headers=headers).status_code == 401


def test_tampered_expired_and_wrong_algorithm_tokens_are_rejected(api: Api) -> None:
    good = api.token("alice")
    assert (
        api.client.get(
            "/api/cases", headers={"Authorization": f"Bearer {good[:-3]}xyz"}
        ).status_code
        == 401
    )
    expired = jwt.encode(
        {"sub": "alice", "role": "analyst", "exp": datetime.now(UTC) - timedelta(minutes=1)},
        SECRET,
        algorithm="HS256",
    )
    assert (
        api.client.get("/api/cases", headers={"Authorization": f"Bearer {expired}"}).status_code
        == 401
    )
    other_key = jwt.encode(
        {"sub": "alice", "role": "analyst", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "another-secret-key-of-sufficient-length-000000",
        algorithm="HS256",
    )
    assert (
        api.client.get("/api/cases", headers={"Authorization": f"Bearer {other_key}"}).status_code
        == 401
    )
    none_alg = jwt.encode(
        {"sub": "alice", "role": "analyst", "exp": datetime.now(UTC) + timedelta(hours=1)},
        None,
        algorithm="none",
    )
    assert (
        api.client.get("/api/cases", headers={"Authorization": f"Bearer {none_alg}"}).status_code
        == 401
    )


def test_permission_matrix(api: Api, busy_customer: str) -> None:
    analyst, auditor, admin = api.headers("alice"), api.headers("audrey"), api.headers("root")
    assert api.client.get("/api/cases", headers=auditor).status_code == 200
    assert api.client.get("/api/cases", headers=admin).status_code == 200
    body = {"customer_id": busy_customer}
    assert api.client.post("/api/cases", json=body, headers=auditor).status_code == 403
    assert api.client.post("/api/cases", json=body, headers=admin).status_code == 403
    assert api.client.get("/api/candidates", headers=auditor).status_code == 403
    new_user = {"username": "dave", "password": PASSWORD, "role": "analyst"}
    assert api.client.post("/api/admin/users", json=new_user, headers=analyst).status_code == 403
    assert api.client.post("/api/admin/users", json=new_user, headers=auditor).status_code == 403
    assert api.client.post("/api/admin/users", json=new_user, headers=admin).status_code == 201
    assert api.client.post("/api/admin/users", json=new_user, headers=admin).status_code == 409
    assert api.token("dave")


def test_user_creation_validates_input(api: Api) -> None:
    admin = api.headers("root")
    for bad in (
        {"username": "Bad Name", "password": PASSWORD, "role": "analyst"},
        {"username": "okname", "password": "short", "role": "analyst"},
        {"username": "okname", "password": PASSWORD, "role": "superuser"},
        {"username": "okname", "password": PASSWORD, "role": "analyst", "extra": 1},
    ):
        assert api.client.post("/api/admin/users", json=bad, headers=admin).status_code == 422


def test_security_headers_and_cors(api: Api) -> None:
    r = api.client.get("/api/auth/me")
    assert (
        r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    )
    assert r.headers["cache-control"] == "no-store"
    ok = api.client.options(
        "/api/cases",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = api.client.options(
        "/api/cases",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in bad.headers
    no_delete = api.client.options(
        "/api/cases",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"},
    )
    assert "DELETE" not in no_delete.headers.get("access-control-allow-methods", "")
