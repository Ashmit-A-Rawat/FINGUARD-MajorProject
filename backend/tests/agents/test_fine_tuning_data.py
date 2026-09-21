import json

import pytest

from agents.context import AgentContext
from agents.coordinator.workflow import CaseWorkflow
from backend.app.schemas.domain import Decision
from guardrails.policy_checks import PolicyConfig, engine_floor
from guardrails.self_critique import SelfCritique
from llm.fine_tuning.dataset import (
    INJECTIONS_EVAL,
    INJECTIONS_TRAIN,
    CaseRecord,
    render,
    stable_nonce,
    target_json,
    training_examples,
)
from llm.fine_tuning.teacher import teacher_investigation
from llm.schemas import InvestigationOutput


def evidence_for(ctx: AgentContext, customer: str, tx_id: str) -> tuple[list, list]:  # type: ignore[type-arg]
    wf = CaseWorkflow(ctx)
    state = wf.run(wf.create_case(customer, [tx_id]))
    return state.evidence, state.knowledge_chunks


def sample_cases(ctx: AgentContext, n: int = 60) -> list[tuple[str, str]]:
    store, out = ctx.store, []
    for customer, txs in store.transactions_by_customer.items():
        if len(txs) > 12:
            out.append((customer, txs[len(txs) // 2].transaction_id))
        if len(out) >= n:
            break
    return out


def test_every_teacher_target_passes_our_own_evidence_validator(ctx: AgentContext) -> None:
    """If the training targets contained unsupported claims we would be teaching the model to fabricate."""
    critique = SelfCritique()
    unsupported, supported, decisions = [], 0, {"CLEAR": 0, "REVIEW": 0}
    for customer, tx_id in sample_cases(ctx):
        evidence, chunks = evidence_for(ctx, customer, tx_id)
        target = teacher_investigation(evidence)
        report = critique.review(evidence, target, chunks)
        supported += report.counts["supported"]
        unsupported += [c for c in report.claim_checks if c.status == "unsupported"]
        decisions[target.recommended_action.value] += 1
        assert report.summary_grounded, (customer, tx_id, report.summary_reason)
        assert not [f for f in report.policy.flags if f.is_violation], report.policy.flags
    assert not unsupported, [(c.claim, c.reason) for c in unsupported[:3]]
    assert supported > 100
    assert decisions["CLEAR"] > 0 and decisions["REVIEW"] > 0  # both classes occur in the sample


def test_teacher_decision_follows_the_engine_floor_and_never_escalates(ctx: AgentContext) -> None:
    for customer, tx_id in sample_cases(ctx, 40):
        evidence, _ = evidence_for(ctx, customer, tx_id)
        target = teacher_investigation(evidence)
        floor, _ = engine_floor(evidence, PolicyConfig())
        assert target.recommended_action == (Decision.REVIEW if floor else Decision.CLEAR)
        assert target.recommended_action.value != "ESCALATE"


def test_teacher_output_round_trips_the_schema_and_is_deterministic(ctx: AgentContext) -> None:
    customer, tx_id = sample_cases(ctx, 1)[0]
    evidence, _ = evidence_for(ctx, customer, tx_id)
    a, b = teacher_investigation(evidence), teacher_investigation(evidence)
    assert a == b
    assert InvestigationOutput.model_validate_json(a.model_dump_json()) == a
    assert all(
        i.startswith(("E:", "K:")) for f in a.findings for i in f.evidence_ids
    )  # prompt's convention


def test_an_injected_memo_forces_review_and_is_mentioned_as_data(ctx: AgentContext) -> None:
    customer, tx_id = next(
        (c, t)
        for c, t in sample_cases(ctx)
        if not engine_floor(evidence_for(ctx, c, t)[0], PolicyConfig())[0]
    )
    evidence, _ = evidence_for(ctx, customer, tx_id)
    clean = teacher_investigation(evidence)
    assert clean.recommended_action == Decision.CLEAR
    attacked = [
        e.model_copy(update={"payload": {**e.payload, "payment_memo": INJECTIONS_TRAIN[0]}})
        if e.evidence_id == tx_id
        else e
        for e in evidence
    ]
    target = teacher_investigation(attacked)
    assert target.recommended_action == Decision.REVIEW
    assert any("instruction-like" in u for u in target.uncertainties)
    assert INJECTIONS_TRAIN[0] not in json.dumps(
        target.model_dump(mode="json")
    )  # the instruction is not repeated


def test_train_and_eval_injection_wordings_are_disjoint() -> None:
    assert not set(INJECTIONS_TRAIN) & set(INJECTIONS_EVAL)


def test_rendered_prompts_use_the_deployed_builder_and_fixed_nonces_are_reproducible(
    ctx: AgentContext,
) -> None:
    customer, tx_id = sample_cases(ctx, 1)[0]
    evidence, chunks = evidence_for(ctx, customer, tx_id)
    record = CaseRecord(
        case_id="c1",
        category="clean",
        truth_problem=False,
        truth_type="none",
        evidence=[e.model_dump(mode="json") for e in evidence],
        chunks=[c.model_dump(mode="json") for c in chunks],
    )
    nonce = stable_nonce("c1")
    with_rag, without = render(record, True, nonce), render(record, False, nonce)
    assert with_rag == render(record, True, nonce)
    assert (
        "[K:" in with_rag[1].content
        and "[K:" not in without[1].content
        and "(none provided)" in without[1].content
    )
    assert f"<<<EVIDENCE {nonce}>>>" in with_rag[1].content
    assert json.loads(target_json(record))["recommended_action"] in ("CLEAR", "REVIEW")


def test_training_examples_vary_rag_and_nonce_but_are_reproducible_for_a_seed(
    ctx: AgentContext,
) -> None:
    records = []
    for i, (customer, tx_id) in enumerate(sample_cases(ctx, 30)):
        evidence, chunks = evidence_for(ctx, customer, tx_id)
        records.append(
            CaseRecord(
                case_id=f"c{i}",
                category="clean",
                truth_problem=False,
                truth_type="none",
                evidence=[e.model_dump(mode="json") for e in evidence],
                chunks=[c.model_dump(mode="json") for c in chunks],
            )
        )
    a, b = training_examples(records, seed=1), training_examples(records, seed=1)
    assert [m[1].content for m, _ in a] == [m[1].content for m, _ in b]
    assert [m[1].content for m, _ in a] != [
        m[1].content for m, _ in training_examples(records, seed=2)
    ]
    rag_share = sum("[K:" in m[1].content for m, _ in a) / len(a)
    assert 0.2 < rag_share < 0.8


@pytest.mark.parametrize("phrase", INJECTIONS_TRAIN + INJECTIONS_EVAL)
def test_evidence_tripwire_status_of_each_wording_is_known(phrase: str) -> None:
    from datetime import datetime

    from backend.app.schemas.domain import Evidence, EvidenceSource
    from guardrails.policy_checks import suspicious_evidence

    e = Evidence(
        evidence_id="TXN-1",
        source=EvidenceSource.TRANSACTION,
        description="t",
        payload={"payment_memo": phrase},
        created_at=datetime(2025, 1, 1),
    )
    flagged = bool(suspicious_evidence([e]))
    # INJECTIONS_EVAL[2] was written to evade the ORIGINAL regex tripwire. The improved patterns
    # (EXP-ADV-02, tuned on the v1 development set) catch it. It still tests the model: the tripwire
    # only adds an engine-floor reason and never changes the prompt the model sees.
    assert flagged, phrase
