from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint_reports_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "fin-guard"


def test_repository_skeleton_importable() -> None:
    import importlib

    for module in ("kyc", "anomaly_detection", "reconciliation", "guardrails", "agents"):
        importlib.import_module(module)
