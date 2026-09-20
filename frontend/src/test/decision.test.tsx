import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Api } from "../api";
import { decisionState } from "../components/decisionRules";
import { DecisionPanel } from "../components/panels/DecisionPanel";
import { analyst, auditor, detail } from "./fixtures";

describe("decisionState", () => {
  it("is read-only for auditors and for closed or running cases", () => {
    expect(decisionState(detail(), auditor, "REVIEW", "x", false).mode).toBe("read-only");
    expect(decisionState(detail({}, { status: "closed" }), analyst, "REVIEW", "x", false).mode).toBe("read-only");
    expect(decisionState(detail({}, { status: "case_created" }), analyst, "REVIEW", "x", false).mode).toBe("read-only");
  });

  it("requires a reason", () => {
    expect(decisionState(detail(), analyst, "REVIEW", "  ", false).canSubmit).toBe(false);
    expect(decisionState(detail(), analyst, "REVIEW", "because", false).canSubmit).toBe(true);
  });

  it("makes CLEAR against a REVIEW advisory require an explicit acknowledgement", () => {
    const d = detail();
    expect(decisionState(d, analyst, "CLEAR", "explained", false)).toMatchObject({ needsOverride: true, canSubmit: false });
    expect(decisionState(d, analyst, "CLEAR", "explained", true)).toMatchObject({ needsOverride: true, canSubmit: true });
    expect(decisionState(d, analyst, "REVIEW", "x", false).needsOverride).toBe(false);
  });

  it("needs no override when the advisory is already CLEAR", () => {
    const d = detail();
    d.report!.advisory_decision = "CLEAR";
    expect(decisionState(d, analyst, "CLEAR", "ok", false)).toMatchObject({ needsOverride: false, canSubmit: true });
  });

  it("applies the four-eyes rule to a pending escalation", () => {
    const pending = detail({ pending_escalation: { proposed_by: "alice", reason: "structuring" } });
    expect(decisionState(pending, analyst, "REVIEW", "", false).mode).toBe("waiting"); // alice cannot confirm her own
    const bob = { ...analyst, username: "bob" };
    expect(decisionState(pending, bob, "REVIEW", "agree", false)).toMatchObject({ mode: "confirm-escalation", canSubmit: true });
    expect(decisionState(pending, { ...analyst, username: "ALICE" }, "REVIEW", "x", false).mode).toBe("waiting");
  });
});

function fakeApi(): Api {
  return { signOff: vi.fn().mockResolvedValue(detail({}, { status: "closed" })), confirmEscalation: vi.fn().mockResolvedValue(detail()), rejectEscalation: vi.fn().mockResolvedValue(detail()) } as unknown as Api;
}

describe("DecisionPanel", () => {
  it("blocks CLEAR until the override is acknowledged, then sends the acknowledgement", async () => {
    const api = fakeApi();
    const onUpdated = vi.fn();
    render(<DecisionPanel detail={detail()} session={analyst} api={api} onUpdated={onUpdated} />);
    await userEvent.click(screen.getByLabelText("CLEAR"));
    await userEvent.type(screen.getByLabelText(/reason/i), "Customer explained the payment");
    const submit = screen.getByRole("button", { name: /sign off as clear/i });
    expect(submit).toBeDisabled();
    await userEvent.click(screen.getByRole("checkbox"));
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    await waitFor(() => expect(api.signOff).toHaveBeenCalledWith("CASE-CUST-1-TXN-1", "CLEAR", "Customer explained the payment", true));
    expect(onUpdated).toHaveBeenCalled();
  });

  it("labels ESCALATE as a proposal that needs a second reviewer", async () => {
    render(<DecisionPanel detail={detail()} session={analyst} api={fakeApi()} onUpdated={vi.fn()} />);
    await userEvent.click(screen.getByLabelText("ESCALATE"));
    expect(screen.getByText(/second, different analyst/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /propose escalation/i })).toBeInTheDocument();
  });

  it("shows read-only guidance to an auditor and offers no decision buttons", () => {
    render(<DecisionPanel detail={detail()} session={auditor} api={fakeApi()} onUpdated={vi.fn()} />);
    expect(screen.getByText(/only analysts can decide/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /sign off/i })).toBeNull();
  });

  it("lets a different analyst confirm or reject a pending escalation", async () => {
    const api = fakeApi();
    const pending = detail({ pending_escalation: { proposed_by: "alice", reason: "structuring" } });
    render(<DecisionPanel detail={pending} session={{ ...analyst, username: "bob" }} api={api} onUpdated={vi.fn()} />);
    await userEvent.type(screen.getByLabelText(/reason/i), "Agree");
    await userEvent.click(screen.getByRole("button", { name: /confirm escalation/i }));
    await waitFor(() => expect(api.confirmEscalation).toHaveBeenCalledWith("CASE-CUST-1-TXN-1", "Agree"));
  });

  it("tells the proposer to wait instead of letting them confirm", () => {
    const pending = detail({ pending_escalation: { proposed_by: "alice", reason: "structuring" } });
    render(<DecisionPanel detail={pending} session={analyst} api={fakeApi()} onUpdated={vi.fn()} />);
    expect(screen.getByText(/you proposed this escalation/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /confirm escalation/i })).toBeNull();
  });

  it("surfaces a server error message", async () => {
    const { ApiError } = await import("../api");
    const api = fakeApi();
    (api.signOff as ReturnType<typeof vi.fn>).mockRejectedValue(new ApiError(409, "the advisory decision is REVIEW"));
    render(<DecisionPanel detail={detail()} session={analyst} api={api} onUpdated={vi.fn()} />);
    await userEvent.type(screen.getByLabelText(/reason/i), "x");
    await userEvent.click(screen.getByRole("button", { name: /sign off as review/i }));
    expect(await screen.findByText(/the advisory decision is REVIEW/)).toBeInTheDocument();
  });

  it("shows who closed the case and any second reviewer", () => {
    const closed = detail({ sign_off: { reviewer: "alice", second_reviewer: "bob", decision: "ESCALATE", reason: "Structuring", at: "2025-07-01T10:00:00" } }, { status: "closed" });
    render(<DecisionPanel detail={closed} session={analyst} api={fakeApi()} onUpdated={vi.fn()} />);
    expect(screen.getByText(/confirmed by bob/)).toBeInTheDocument();
    expect(screen.getByText(/this case is closed/i)).toBeInTheDocument();
  });
});
