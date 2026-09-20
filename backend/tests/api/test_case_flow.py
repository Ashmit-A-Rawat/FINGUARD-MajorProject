import json
from pathlib import Path

import pytest
from api_helpers import Api
from sqlalchemy import text

from agents.audit import AuditEvent
from agents.context import AgentContext
from backend.app.core.config import Settings
from backend.app.database.models import AuditImmutableError, AuditRow, CaseRow
from backend.app.main import create_app
from backend.app.services.audit_chain import GENESIS, append_events, verify_chain
from backend.app.services.engine import EngineService


def open_case(api: Api, customer: str, user: str = "alice", tx: list[str] | None = None) -> str:
    body: dict[str, object] = {"customer_id": customer}
    if tx:
        body["transaction_ids"] = tx
    r = api.client.post("/api/cases", json=body, headers=api.headers(user))
    assert r.status_code == 202, r.text
    case_id = str(r.json()["case_id"])
    assert api.wait_for_job(case_id) == "done"
    return case_id


def test_health_reports_engine_state(api: Api) -> None:
    body = api.client.get("/health").json()
    assert body["engine_ready"] is True and body["status"] == "ok"


def test_create_case_runs_the_workflow_and_persists_everything(
    api: Api, busy_customer: str
) -> None:
    case_id = open_case(api, busy_customer)
    detail = api.client.get(f"/api/cases/{case_id}", headers=api.headers("alice")).json()
    s = detail["summary"]
    assert s["status"] == "human_review" and s["job_state"] == "done" and s["created_by"] == "alice"
    assert s["is_mock"] is True and s["validated"] is False  # mock output is never validated
    assert detail["report"]["engine_facts"] and detail["evidence"] and detail["knowledge_chunks"]
    assert detail["investigation"]["summary"].startswith("[MOCK OUTPUT")
    assert detail["review"]["critique"]["claim_checks"] is not None
    assert any(item["is_focus"] for item in detail["timeline"])
    assert detail["customer"]["kyc_status"] and detail["customer"]["details"]["name"]
    listed = api.client.get("/api/cases", headers=api.headers("audrey")).json()
    assert listed["total"] == 1 and listed["items"][0]["case_id"] == case_id
    assert (
        api.client.get("/api/cases?status=closed", headers=api.headers("audrey")).json()["total"]
        == 0
    )


def test_only_analysts_see_personal_details(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    name = api.engine.context.store.customers[busy_customer].customer.name  # type: ignore[union-attr]
    analyst = api.client.get(f"/api/cases/{case_id}", headers=api.headers("alice"))
    auditor = api.client.get(f"/api/cases/{case_id}", headers=api.headers("audrey"))
    admin = api.client.get(f"/api/cases/{case_id}", headers=api.headers("root"))
    assert name in analyst.text
    assert name not in auditor.text and "details" not in auditor.json()["customer"]
    assert name not in admin.text and "details" not in admin.json()["customer"]


def test_every_access_is_written_to_the_audit_trail(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    api.client.get(f"/api/cases/{case_id}", headers=api.headers("alice"))
    api.client.get(f"/api/cases/{case_id}", headers=api.headers("audrey"))
    audit = api.client.get(f"/api/cases/{case_id}/audit", headers=api.headers("audrey")).json()
    actions = [(e["actor"], e["action"]) for e in audit["events"]]
    assert ("human:alice", "request_case") in actions
    assert ("human:alice", "view_case") in actions and (
        "human:alice",
        "view_customer_details",
    ) in actions
    assert ("human:audrey", "view_case") in actions
    assert ("human:audrey", "view_audit_trail") in actions  # a read is recorded before it is served
    again = api.client.get(f"/api/cases/{case_id}/audit", headers=api.headers("root")).json()
    views = [e["actor"] for e in again["events"] if e["action"] == "view_audit_trail"]
    assert views == ["human:audrey", "human:root"]
    assert [e["event_id"] for e in again["events"]] == list(range(1, len(again["events"]) + 1))
    assert audit["chain"]["ok"] is True
    steps = [
        e["action"]
        for e in audit["events"]
        if e["actor"]
        in (
            "coordinator",
            "auditor",
            "reconciliation",
            "investigator",
            "reviewer",
            "report_generator",
        )
    ]
    assert "prepare_data" in steps and "generate_report" in steps


def test_invalid_case_requests(api: Api, busy_customer: str, engine_ctx: AgentContext) -> None:
    h = api.headers("alice")
    assert (
        api.client.post("/api/cases", json={"customer_id": "CUST-NOPE"}, headers=h).status_code
        == 404
    )
    other = next(c for c in engine_ctx.store.transactions_by_customer if c != busy_customer)
    foreign = engine_ctx.store.transactions_by_customer[other][0].transaction_id
    r = api.client.post(
        "/api/cases", json={"customer_id": busy_customer, "transaction_ids": [foreign]}, headers=h
    )
    assert r.status_code == 422
    for bad in (
        {"customer_id": "../etc/passwd"},
        {"customer_id": busy_customer, "context_days": 0},
        {"customer_id": busy_customer, "transaction_ids": ["x"] * 21},
        {"customer_id": busy_customer, "role": "admin"},
    ):
        assert api.client.post("/api/cases", json=bad, headers=h).status_code == 422
    assert api.client.get("/api/cases/bad%20id", headers=h).status_code == 422
    assert api.client.get("/api/cases/CASE-DOES-NOT-EXIST", headers=h).status_code == 404
    open_case(api, busy_customer)
    assert (
        api.client.post("/api/cases", json={"customer_id": busy_customer}, headers=h).status_code
        == 409
    )


def test_creating_a_case_before_the_engine_is_ready_is_a_clear_503(
    tmp_path: Path, busy_customer: str
) -> None:
    from datetime import datetime

    from fastapi.testclient import TestClient

    from backend.app.core.security import Role, hash_password
    from backend.app.database.models import UserRow

    settings = Settings(
        app_env="test",
        api_secret_key="x" * 40,
        database_url=f"sqlite:///{tmp_path / 'e.db'}",
        engine_enabled=False,
    )
    app = create_app(settings, EngineService())  # engine object exists but nothing loaded
    with TestClient(app) as client:
        with app.state.session_factory() as s:
            s.add(
                UserRow(
                    username="alice",
                    password_hash=hash_password("a long enough pass"),
                    role=Role.ANALYST.value,
                    active=True,
                    created_at=datetime.utcnow(),
                )
            )
            s.commit()
        token = client.post(
            "/api/auth/login", json={"username": "alice", "password": "a long enough pass"}
        ).json()["access_token"]
        r = client.post(
            "/api/cases",
            json={"customer_id": busy_customer},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 503 and "loading" in r.json()["detail"]
        assert client.get("/health").json()["engine_ready"] is False


def test_candidates_come_from_the_held_out_period(api: Api, engine_ctx: AgentContext) -> None:
    items = api.client.get("/api/candidates?limit=10", headers=api.headers("alice")).json()
    assert len(items) == 10 and not any(i["has_case"] for i in items)
    cutoff = engine_ctx.anomaly.trained_until.to_pydatetime()
    assert all(i["timestamp"] >= cutoff.isoformat() for i in items)
    open_case(api, items[0]["customer_id"], tx=[items[0]["transaction_id"]])
    refreshed = api.client.get("/api/candidates?limit=10", headers=api.headers("alice")).json()
    assert refreshed[0]["has_case"] is True


# ---------------- sign-off ----------------
def test_sign_off_uses_the_authenticated_identity_and_closes_the_case(
    api: Api, busy_customer: str
) -> None:
    case_id = open_case(api, busy_customer)
    h = api.headers("alice")
    advisory = api.client.get(f"/api/cases/{case_id}", headers=h).json()["report"][
        "advisory_decision"
    ]
    body = {"decision": "REVIEW", "reason": "Needs a second document.", "reviewer": "someone-else"}
    assert (
        api.client.post(f"/api/cases/{case_id}/sign-off", json=body, headers=h).status_code == 422
    )  # no spoofed identity
    body.pop("reviewer")
    r = api.client.post(f"/api/cases/{case_id}/sign-off", json=body, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["sign_off"]["reviewer"] == "alice" and r.json()["summary"]["status"] == "closed"
    assert advisory in (None, "REVIEW", "ESCALATE", "CLEAR")
    again = api.client.post(f"/api/cases/{case_id}/sign-off", json=body, headers=h)
    assert again.status_code == 409


def test_choosing_clear_against_the_advisory_needs_an_audited_override(
    api: Api, busy_customer: str
) -> None:
    case_id = open_case(api, busy_customer)
    h = api.headers("alice")
    advisory = api.client.get(f"/api/cases/{case_id}", headers=h).json()["report"][
        "advisory_decision"
    ]
    assert advisory == "REVIEW"  # the mock proposes REVIEW
    body = {"decision": "CLEAR", "reason": "Customer explained the payment."}
    blocked = api.client.post(f"/api/cases/{case_id}/sign-off", json=body, headers=h)
    assert (
        blocked.status_code == 409 and "acknowledge_advisory_override" in blocked.json()["detail"]
    )
    ok = api.client.post(
        f"/api/cases/{case_id}/sign-off",
        json={**body, "acknowledge_advisory_override": True},
        headers=h,
    )
    assert ok.status_code == 200 and ok.json()["sign_off"]["decision"] == "CLEAR"
    events = api.client.get(f"/api/cases/{case_id}/audit", headers=h).json()["events"]
    override = next(e for e in events if e["action"] == "advisory_override_acknowledged")
    assert override["actor"] == "human:alice" and override["details"] == {
        "advisory": "REVIEW",
        "chosen": "CLEAR",
    }


def test_escalation_needs_a_second_different_authenticated_reviewer(
    api: Api, busy_customer: str
) -> None:
    case_id = open_case(api, busy_customer)
    alice, bob = api.headers("alice"), api.headers("bob")
    proposed = api.client.post(
        f"/api/cases/{case_id}/sign-off",
        json={"decision": "ESCALATE", "reason": "Looks like structuring."},
        headers=alice,
    )
    assert proposed.status_code == 200
    assert (
        proposed.json()["summary"]["status"] == "human_review"
        and proposed.json()["pending_escalation"]["proposed_by"] == "alice"
    )
    assert (
        api.client.post(
            f"/api/cases/{case_id}/sign-off",
            json={"decision": "REVIEW", "reason": "x"},
            headers=alice,
        ).status_code
        == 409
    )
    self_confirm = api.client.post(
        f"/api/cases/{case_id}/sign-off/confirm",
        json={"reason": "I agree with myself"},
        headers=alice,
    )
    assert self_confirm.status_code == 403 and "different person" in self_confirm.json()["detail"]
    assert (
        api.client.post(
            f"/api/cases/{case_id}/sign-off/confirm",
            json={"reason": "ok"},
            headers=api.headers("audrey"),
        ).status_code
        == 403
    )
    done = api.client.post(
        f"/api/cases/{case_id}/sign-off/confirm",
        json={"reason": "Confirmed after review."},
        headers=bob,
    )
    assert done.status_code == 200
    so = done.json()["sign_off"]
    assert (so["reviewer"], so["second_reviewer"], so["decision"]) == ("alice", "bob", "ESCALATE")
    assert (
        done.json()["summary"]["status"] == "closed" and done.json()["pending_escalation"] is None
    )
    actions = [
        (e["actor"], e["action"])
        for e in api.client.get(f"/api/cases/{case_id}/audit", headers=alice).json()["events"]
    ]
    assert ("human:alice", "escalation_proposed") in actions and (
        "human:bob",
        "escalation_confirmed",
    ) in actions
    assert ("human:alice", "sign_off") in actions


def test_second_reviewer_can_reject_an_escalation_and_the_case_stays_open(
    api: Api, busy_customer: str
) -> None:
    case_id = open_case(api, busy_customer)
    api.client.post(
        f"/api/cases/{case_id}/sign-off",
        json={"decision": "ESCALATE", "reason": "Suspicious."},
        headers=api.headers("alice"),
    )
    assert (
        api.client.post(
            f"/api/cases/{case_id}/sign-off/reject",
            json={"reason": "no"},
            headers=api.headers("alice"),
        ).status_code
        == 403
    )
    r = api.client.post(
        f"/api/cases/{case_id}/sign-off/reject",
        json={"reason": "Evidence is weak."},
        headers=api.headers("bob"),
    )
    assert (
        r.status_code == 200
        and r.json()["summary"]["status"] == "human_review"
        and r.json()["pending_escalation"] is None
    )
    assert (
        api.client.post(
            f"/api/cases/{case_id}/sign-off/confirm",
            json={"reason": "x"},
            headers=api.headers("bob"),
        ).status_code
        == 409
    )


def test_auditors_and_admins_cannot_decide_cases(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    body = {"decision": "REVIEW", "reason": "x"}
    for user in ("audrey", "root"):
        assert (
            api.client.post(
                f"/api/cases/{case_id}/sign-off", json=body, headers=api.headers(user)
            ).status_code
            == 403
        )


def test_there_is_no_endpoint_that_can_take_a_financial_action(api: Api) -> None:
    paths = set(api.app.openapi()["paths"])  # type: ignore[attr-defined]
    forbidden = ("freeze", "block", "reverse", "refund", "transfer", "close_account", "contact")
    assert not [p for p in paths if any(word in p.lower() for word in forbidden)]


# ---------------- audit-trail integrity ----------------
def test_tampering_with_a_stored_event_is_detected(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    h = api.headers("audrey")
    assert api.client.get(f"/api/cases/{case_id}/audit", headers=h).json()["chain"]["ok"] is True
    with api.app.state.session_factory() as s:  # type: ignore[attr-defined]
        # bypass the ORM guard, as an attacker with direct database access would
        s.execute(
            text("UPDATE audit_events SET actor = 'human:mallory' WHERE case_id = :c AND seq = 3"),
            {"c": case_id},
        )
        s.commit()
    chain = api.client.get(f"/api/cases/{case_id}/audit", headers=h).json()["chain"]
    assert chain["ok"] is False and chain["first_bad_seq"] == 3 and "content" in chain["reason"]


def test_deleting_or_reordering_events_is_detected(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    with api.app.state.session_factory() as s:  # type: ignore[attr-defined]
        s.execute(text("DELETE FROM audit_events WHERE case_id = :c AND seq = 2"), {"c": case_id})
        s.commit()
        report = verify_chain(s, case_id)
    assert not report.ok and report.first_bad_seq == 3 and "gap" in report.reason


def test_the_orm_refuses_to_update_or_delete_audit_events(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    with api.app.state.session_factory() as s:  # type: ignore[attr-defined]
        row = s.query(AuditRow).filter_by(case_id=case_id, seq=1).one()
        row.actor = "human:mallory"
        with pytest.raises(AuditImmutableError):
            s.flush()
        s.rollback()
        row = s.query(AuditRow).filter_by(case_id=case_id, seq=1).one()
        s.delete(row)
        with pytest.raises(AuditImmutableError):
            s.flush()


def test_interleaved_writers_never_drop_or_duplicate_events(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    with api.app.state.session_factory() as s:  # type: ignore[attr-defined]
        before = s.query(AuditRow).filter_by(case_id=case_id).count()
        row = s.get(CaseRow, case_id)
        assert row is not None
        template = AuditEvent(
            event_id=1,
            timestamp=row.created_at,
            case_id=case_id,
            request_id="r",
            actor="human:x",
            action="a",
            from_status=None,
            to_status=None,
            duration_ms=0.0,
            ok=True,
        )
        # two writers each believe their event is number 1: the database assigns the real order
        append_events(s, case_id, [template.model_copy(update={"action": "first"})])
        append_events(s, case_id, [template.model_copy(update={"action": "second"})])
        s.commit()
        rows = s.query(AuditRow).filter_by(case_id=case_id).order_by(AuditRow.seq).all()
        assert len(rows) == before + 2 and [r.seq for r in rows] == list(range(1, before + 3))
        assert [r.action for r in rows[-2:]] == ["first", "second"] and verify_chain(s, case_id).ok
        assert rows[0].prev_hash == GENESIS


def test_state_and_audit_survive_a_restart(api: Api, busy_customer: str) -> None:
    case_id = open_case(api, busy_customer)
    url = str(api.app.state.settings.database_url)  # type: ignore[attr-defined]
    from fastapi.testclient import TestClient

    second = create_app(
        Settings(app_env="test", api_secret_key="y" * 40, database_url=url, engine_enabled=False)
    )
    with TestClient(second) as client:
        with second.state.session_factory() as s:
            assert s.get(CaseRow, case_id) is not None
            assert verify_chain(s, case_id).ok
            row = s.get(CaseRow, case_id)
            assert row is not None and json.loads(row.state_json)["case_id"] == case_id
        assert client.get("/health").json()["engine_ready"] is False
