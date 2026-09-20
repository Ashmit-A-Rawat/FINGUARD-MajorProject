import { describe, expect, it, vi } from "vitest";
import { Api, ApiError } from "../api";

const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

describe("Api client", () => {
  it("sends the bearer token and never puts it in the URL", async () => {
    const fetcher = vi.fn().mockImplementation(() => json({ items: [], total: 0 }));
    await new Api("secret-token", undefined, fetcher).listCases("closed");
    const [url, init] = fetcher.mock.calls[0];
    expect(url).toBe("/api/cases?limit=100&status=closed");
    expect(String(url)).not.toContain("secret-token");
    expect((init.headers as Headers).get("Authorization")).toBe("Bearer secret-token");
  });

  it("turns error responses into ApiError with the server's message", async () => {
    const fetcher = vi.fn().mockImplementation(() => json({ detail: "the advisory decision is REVIEW" }, 409));
    await expect(new Api("t", undefined, fetcher).signOff("C", "CLEAR", "r", false)).rejects.toMatchObject({ status: 409, message: "the advisory decision is REVIEW" });
  });

  it("signs the user out on 401 but not on other errors", async () => {
    const onUnauthorized = vi.fn();
    const f401 = vi.fn().mockImplementation(() => json({ detail: "no" }, 401));
    await expect(new Api("t", onUnauthorized, f401).getCase("C")).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).toHaveBeenCalledOnce();
    const f403 = vi.fn().mockImplementation(() => json({ detail: "no" }, 403));
    await expect(new Api("t", onUnauthorized, f403).getCase("C")).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it("encodes case ids in paths", async () => {
    const fetcher = vi.fn().mockImplementation(() => json({}));
    await new Api("t", undefined, fetcher).getCase("CASE/../x y");
    expect(fetcher.mock.calls[0][0]).toBe("/api/cases/CASE%2F..%2Fx%20y");
  });

  it("logs in without sending an Authorization header and returns the session", async () => {
    const fetcher = vi.fn().mockImplementation(() => json({ access_token: "abc", username: "alice", role: "analyst" }));
    const s = await Api.login("alice", "pw", fetcher);
    expect(s).toEqual({ token: "abc", username: "alice", role: "analyst" });
    expect((fetcher.mock.calls[0][1].headers as Headers).has("Authorization")).toBe(false);
  });

  it("sends the override flag with a sign-off", async () => {
    const fetcher = vi.fn().mockImplementation(() => json({}));
    await new Api("t", undefined, fetcher).signOff("C", "CLEAR", "why", true);
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ decision: "CLEAR", reason: "why", acknowledge_advisory_override: true });
  });
});
