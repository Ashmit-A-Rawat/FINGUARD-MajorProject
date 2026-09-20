import time
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.security import Role, hash_password
from backend.app.database.models import UserRow
from backend.app.services.engine import EngineService


class HashingEmbedder:
    """Test-only deterministic embedder. NOT a real dense model."""

    name = "test-hashing-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normalised: np.ndarray = out / np.maximum(norms, 1e-9)
        return normalised


class FakeClock:
    """Deterministic clock: each call advances one second."""

    def __init__(self) -> None:
        self._now = datetime(2025, 7, 1, 9, 0, 0)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


SECRET = "test-secret-key-that-is-definitely-long-enough-1234567890"
PASSWORD = "correct horse battery staple"


class Api:
    def __init__(self, client: TestClient, engine: EngineService | None) -> None:
        self.client, self.engine = client, engine
        self.app = client.app

    def add_user(
        self, username: str, role: Role, password: str = PASSWORD, active: bool = True
    ) -> None:
        with self.app.state.session_factory() as s:  # type: ignore[attr-defined]
            s.add(
                UserRow(
                    username=username,
                    password_hash=hash_password(password),
                    role=role.value,
                    active=active,
                    created_at=datetime.utcnow(),
                )
            )
            s.commit()

    def token(self, username: str, password: str = PASSWORD) -> str:
        r = self.client.post("/api/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return str(r.json()["access_token"])

    def headers(self, username: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(username)}"}

    def wait_for_job(self, case_id: str, timeout: float = 60.0) -> str:
        from backend.app.database.models import CaseRow

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.app.state.session_factory() as s:  # type: ignore[attr-defined]
                row = s.get(CaseRow, case_id)
                if row is not None and row.job_state in ("done", "failed"):
                    return str(row.job_state)
            time.sleep(0.05)
        raise TimeoutError(case_id)
