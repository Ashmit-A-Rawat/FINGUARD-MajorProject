import pytest

from agents.baseline import SingleAgentBaseline
from agents.context import AgentContext
from agents.coordinator.workflow import CaseWorkflow, SignOffError
from agents.reviewer.agent import ReviewerAgent
from agents.state import CaseState
from backend.app.schemas.domain import CaseStatus, Decision
from llm.inference.mock import MOCK_BANNER, MockLLMProvider

SPEC_ORDER = [
    CaseStatus.CASE_CREATED,
    CaseStatus.DATA_READY,
    CaseStatus.KYC_ANALYZED,
    CaseStatus.ANOMALY_ANALYZED,
    CaseStatus.RECONCILED,
    CaseStatus.EVIDENCE_RETRIEVED,
    CaseStatus.INVESTIGATION_GENERATED,
    CaseStatus.SELF_CRITIQUED,
    CaseStatus.HUMAN_REVIEW,
]


def with_llm(ctx: AgentContext, provider: MockLLMProvider) -> AgentContext:
    return AgentContext(**{**ctx.__dict__, "llm": provider})


def test_workflow_follows_the_specified_state_sequence_and_stops_at_human_review(
    ctx: AgentContext, busy_customer: str
) -> None:
    wf = CaseWorkflow(ctx)
    state = wf.run(wf.create_case(busy_customer))
    assert state.status == CaseStatus.HUMAN_REVIEW  # never CLOSED automatically
    steps = [e for e in state.audit_trail if e.action not in ("create_case",)]
    assert [e.from_status for e in steps] == SPEC_ORDER[:-1]
    assert [e.to_status for e in steps] == SPEC_ORDER[1:]
    assert [e.actor for e in steps] == [
        "coordinator",
        "auditor",
        "auditor",
        "reconciliation",
        "investigator",
        "investigator",
        "reviewer",
        "report_generator",
    ]
    assert state.failed_step is None and all(e.ok for e in state.audit_trail)


def test_audit_trail_is_ordered_complete_and_carries_no_personal_data(
    ctx: AgentContext, busy_customer: str
) -> None:
    wf = CaseWorkflow(ctx)
    state = wf.run(wf.create_case(busy_customer, request_id="req-1"))
    events = state.audit_trail
    assert [e.event_id for e in events] == list(range(1, len(events) + 1))
    assert [e.timestamp for e in events] == sorted(e.timestamp for e in events)
    assert {e.request_id for e in events} == {"req-1"} and {e.case_id for e in events} == {
        state.case_id
    }
    name = ctx.store.customers[busy_customer].customer.name
    assert name not in " ".join(str(e.model_dump()) for e in events)
    assert name not in state.report.model_dump_json()  # type: ignore[union-attr]
    llm_event = next(e for e in events if e.action == "investigate")
    assert llm_event.details["is_mock"] is True and llm_event.details["provider"] == "mock"


def test_evidence_is_unique_typed_and_covers_every_engine(
    ctx: AgentContext, busy_customer: str
) -> None:
    wf = CaseWorkflow(ctx)
    state = wf.run(wf.create_case(busy_customer))
    ids = [e.evidence_id for e in state.evidence]
    assert len(ids) == len(set(ids))
    prefixes = {i.split("-")[0] for i in ids}
    assert {"KYC", "KYCM", "TXN", "BEH", "ANOM", "REC"} <= prefixes
    assert state.kyc_findings and state.anomaly_findings and state.reconciliation_results
    assert state.knowledge_chunks and all(
        c.metadata["trust"] == "trusted" for c in state.knowledge_chunks
    )


def test_report_separates_engine_facts_from_model_statements_and_is_honest_about_mock(
    ctx: AgentContext, busy_customer: str
) -> None:
    wf = CaseWorkflow(ctx)
    report = wf.run(wf.create_case(busy_customer)).report
    assert report is not None and report.advisory_only and report.is_synthetic
    assert report.engine_facts and all(i.origin == "engine" for i in report.engine_facts)
    assert all(i.origin == "model" and MOCK_BANNER in i.text for i in report.model_findings)
    assert not report.validated and any("UNVALIDATED" in w for w in report.warnings)
    assert any("MOCK OUTPUT" in w for w in report.warnings)
    assert report.provenance["is_mock"] is True and report.provenance["prompt_version"]
    cited = {i for f in report.engine_facts for i in f.evidence_ids}
    assert cited <= set(report.evidence_index)


def test_a_failing_step_fails_safe_to_human_review_with_a_report(
    ctx: AgentContext, busy_customer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    wf = CaseWorkflow(ctx)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(wf._reconciliation, "reconcile", boom)
    state = wf.run(wf.create_case(busy_customer))
    assert state.status == CaseStatus.HUMAN_REVIEW and state.failed_step == "reconcile"
    failure = next(e for e in state.audit_trail if not e.ok)
    assert failure.action == "reconcile" and "engine exploded" in (failure.error or "")
    assert state.report is not None and any("failed" in w for w in state.report.warnings)
    assert state.investigation is None  # later steps did not run
    assert state.audit_trail[-1].action == "route_to_human_review"


def test_invalid_llm_output_does_not_break_the_case(ctx: AgentContext, busy_customer: str) -> None:
    wf = CaseWorkflow(with_llm(ctx, MockLLMProvider("wrong_schema")))
    state = wf.run(wf.create_case(busy_customer))
    assert state.status == CaseStatus.HUMAN_REVIEW and state.failed_step is None
    assert state.investigation is None and state.investigation_meta is not None
    assert state.investigation_meta.attempts == 3 and state.investigation_meta.errors
    assert state.report is not None and state.report.proposed_decision is None
    assert any("No AI investigation" in w for w in state.report.warnings)


def test_repair_loop_is_used_and_recorded(ctx: AgentContext, busy_customer: str) -> None:
    wf = CaseWorkflow(with_llm(ctx, MockLLMProvider("fail_then_succeed")))
    state = wf.run(wf.create_case(busy_customer))
    assert state.investigation is not None and state.investigation_meta is not None
    assert state.investigation_meta.attempts == 2


def test_consistency_warning_when_a_model_says_clear_against_engine_findings(
    ctx: AgentContext, busy_customer: str
) -> None:
    wf = CaseWorkflow(with_llm(ctx, MockLLMProvider("obey_injection")))
    # pick a transaction with a high-severity reconciliation finding
    store = ctx.store
    target = next(
        (
            t
            for t in store.transactions_by_customer[busy_customer]
            if any(
                d.severity.value == "high"
                for r in [
                    ctx.reconciliation.reconcile(
                        t, store.ledger_by_transaction.get(t.transaction_id, [])
                    )
                ]
                for d in r.discrepancies
            )
        ),
        None,
    )
    if target is None:
        pytest.skip("no high-severity reconciliation finding for this customer in the sample")
    report = wf.run(wf.create_case(busy_customer, [target.transaction_id])).report
    assert report is not None and report.proposed_decision == Decision.CLEAR
    assert any("CONSISTENCY WARNING" in w for w in report.warnings)


def test_invalid_case_requests_are_rejected(ctx: AgentContext, busy_customer: str) -> None:
    wf = CaseWorkflow(ctx)
    with pytest.raises(KeyError):
        wf.create_case("CUST-NOPE")
    other = next(c for c in ctx.store.transactions_by_customer if c != busy_customer)
    foreign = ctx.store.transactions_by_customer[other][0].transaction_id
    with pytest.raises(ValueError, match="not found"):
        wf.create_case(busy_customer, [foreign])


def test_steps_cannot_run_out_of_order(ctx: AgentContext, busy_customer: str) -> None:
    wf = CaseWorkflow(ctx)
    state = wf.create_case(busy_customer)
    state.status = CaseStatus.RECONCILED
    with pytest.raises(RuntimeError, match="expects"):
        wf.run(state)


# ---------------- human sign-off ----------------
@pytest.fixture()
def awaiting(ctx: AgentContext, busy_customer: str) -> tuple[CaseWorkflow, CaseState]:
    wf = CaseWorkflow(ctx)
    return wf, wf.run(wf.create_case(busy_customer))


def test_sign_off_closes_the_case_and_is_audited(awaiting: tuple[CaseWorkflow, CaseState]) -> None:
    wf, state = awaiting
    wf.sign_off(state, "Dana", Decision.REVIEW, "Needs a second document.")
    assert state.status == CaseStatus.CLOSED and state.sign_off is not None
    last = state.audit_trail[-1]
    assert last.actor == "human:Dana" and last.details == {"decision": "REVIEW"}
    with pytest.raises(SignOffError, match="not awaiting"):
        wf.sign_off(state, "Dana", Decision.CLEAR, "again")


@pytest.mark.parametrize(
    ("reviewer", "reason"), [("", "why"), ("  ", "why"), ("Dana", ""), ("Dana", "  ")]
)
def test_sign_off_requires_a_named_reviewer_and_a_reason(
    awaiting: tuple[CaseWorkflow, CaseState], reviewer: str, reason: str
) -> None:
    wf, state = awaiting
    with pytest.raises(SignOffError, match="named reviewer"):
        wf.sign_off(state, reviewer, Decision.CLEAR, reason)
    assert state.status == CaseStatus.HUMAN_REVIEW


@pytest.mark.parametrize("second", [None, "", "dana", "  DANA "])
def test_escalate_needs_a_different_second_reviewer(
    awaiting: tuple[CaseWorkflow, CaseState], second: str | None
) -> None:
    wf, state = awaiting
    with pytest.raises(SignOffError, match="four-eyes"):
        wf.sign_off(state, "Dana", Decision.ESCALATE, "Looks like structuring.", second)
    wf.sign_off(state, "Dana", Decision.ESCALATE, "Looks like structuring.", "Ravi")
    assert (
        state.status == CaseStatus.CLOSED
        and state.sign_off is not None
        and state.sign_off.second_reviewer == "Ravi"
    )


def test_cannot_sign_off_before_review(ctx: AgentContext, busy_customer: str) -> None:
    wf = CaseWorkflow(ctx)
    with pytest.raises(SignOffError):
        wf.sign_off(wf.create_case(busy_customer), "Dana", Decision.CLEAR, "too early")


# ---------------- reviewer plug-in ----------------
def test_reviewer_reports_unvalidated_until_validators_exist_and_pass(
    ctx: AgentContext, busy_customer: str
) -> None:
    from agents.contracts import CheckResult, ReviewInput

    class Always:
        check_id = "always"

        def __init__(self, ok: bool) -> None:
            self.ok = ok

        def validate(self, inp: ReviewInput) -> list[CheckResult]:
            return [CheckResult(check_id=self.check_id, passed=self.ok, detail="x")]

    wf = CaseWorkflow(ctx)
    plain = wf.run(wf.create_case(busy_customer))
    assert plain.review is not None and not plain.review.validated and not plain.review.checks
    for ok in (True, False):
        wf2 = CaseWorkflow(ctx, ReviewerAgent([Always(ok)]))
        state = wf2.run(wf2.create_case(busy_customer))
        assert state.review is not None and state.review.validated is ok
        assert (state.report is not None and state.report.validated) is ok


# ---------------- services and baseline ----------------
def test_anomaly_service_trains_early_and_marks_training_period_scores(
    ctx: AgentContext, busy_customer: str
) -> None:
    txs = ctx.store.transactions_by_customer[busy_customer]
    first, last = (
        ctx.anomaly.score(txs[0].transaction_id),
        ctx.anomaly.score(txs[-1].transaction_id),
    )
    assert first.in_training_period and not last.in_training_period
    assert 0.0 <= last.probability <= 1.0 and len(last.top_drivers) == 3
    assert ctx.anomaly.score(txs[-1].transaction_id) == last  # deterministic


def test_retrieval_queries_are_deterministic_and_mention_rule_ids(ctx: AgentContext) -> None:
    from agents.investigator.queries import DECISION_QUERY, build_queries
    from llm.rag.evidence import reconciliation_evidence

    store = ctx.store
    target = next(
        t
        for txs in store.transactions_by_customer.values()
        for t in txs
        if not store.ledger_by_transaction.get(t.transaction_id)
    )
    result = ctx.reconciliation.reconcile(target, [])
    evidence = reconciliation_evidence(result, ctx.clock())
    queries = build_queries(evidence)
    assert (
        queries == build_queries(evidence)
        and queries[0].startswith("REC-001")
        and queries[-1] == DECISION_QUERY
    )


def test_single_agent_baseline_uses_one_llm_call_and_no_workflow(
    ctx: AgentContext, busy_customer: str
) -> None:
    provider = MockLLMProvider()
    result = SingleAgentBaseline(with_llm(ctx, provider)).run(busy_customer)
    assert provider.calls == 1 and result.investigation is not None and result.meta.is_mock
    assert result.meta.attempts == 1 and result.wall_seconds >= 0
