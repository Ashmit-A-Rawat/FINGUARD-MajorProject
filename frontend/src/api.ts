import type {
  AuditResponse,
  Candidate,
  CaseDetail,
  CaseSummary,
  Decision,
  Session,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Fetcher = typeof fetch;

/** Thin typed client. The token is held by the caller (in memory), never in the URL. */
export class Api {
  constructor(
    private token: string | null,
    private onUnauthorized: () => void = () => undefined,
    private fetcher: Fetcher = (...args) => fetch(...args),
  ) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Content-Type", "application/json");
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    const response = await this.fetcher(`/api${path}`, { ...init, headers });
    if (response.status === 401 && this.token) this.onUnauthorized();
    if (!response.ok) {
      let message = response.statusText;
      try {
        const body = await response.json();
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* keep statusText */
      }
      throw new ApiError(response.status, message);
    }
    return (await response.json()) as T;
  }

  static async login(username: string, password: string, fetcher: Fetcher = (...a) => fetch(...a)): Promise<Session> {
    const api = new Api(null, undefined, fetcher);
    const r = await api.request<{ access_token: string; username: string; role: Session["role"] }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    return { token: r.access_token, username: r.username, role: r.role };
  }

  listCases(status?: string): Promise<{ items: CaseSummary[]; total: number }> {
    return this.request(`/cases?limit=100${status ? `&status=${encodeURIComponent(status)}` : ""}`);
  }
  getCase(id: string): Promise<CaseDetail> {
    return this.request(`/cases/${encodeURIComponent(id)}`);
  }
  createCase(customerId: string, transactionIds?: string[]): Promise<CaseSummary> {
    return this.request("/cases", {
      method: "POST",
      body: JSON.stringify({ customer_id: customerId, transaction_ids: transactionIds ?? null }),
    });
  }
  candidates(): Promise<Candidate[]> {
    return this.request("/candidates?limit=30");
  }
  audit(id: string): Promise<AuditResponse> {
    return this.request(`/cases/${encodeURIComponent(id)}/audit`);
  }
  signOff(id: string, decision: Decision, reason: string, override: boolean): Promise<CaseDetail> {
    return this.request(`/cases/${encodeURIComponent(id)}/sign-off`, {
      method: "POST",
      body: JSON.stringify({ decision, reason, acknowledge_advisory_override: override }),
    });
  }
  confirmEscalation(id: string, reason: string): Promise<CaseDetail> {
    return this.request(`/cases/${encodeURIComponent(id)}/sign-off/confirm`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  }
  rejectEscalation(id: string, reason: string): Promise<CaseDetail> {
    return this.request(`/cases/${encodeURIComponent(id)}/sign-off/reject`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  }
}
