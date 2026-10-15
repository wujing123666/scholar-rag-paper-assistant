import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiFailure, runRequest } from "./client";
import type { PendingRequest } from "./client";

const intent = (): PendingRequest => ({ key: "idempotency_client_test", job_id: null, kind: "answer", query: "test",
  paper_ids: [], dataset_id: "api-test", session_id: "session-test", created_at: Date.now() });
const complete = { job_id: "job-1", kind: "answer", dataset_id: "api-test", state: "completed", error: null,
  result: { status: "refused", claims: [], citations: [], reason: "not_enough_retrieved_chunks", source: "api", verification: { status: "skipped" } } };
const response = (value: unknown, status = 200) => ({ ok: status < 400, status, json: async () => value });
afterEach(() => vi.unstubAllGlobals());
describe("HTTP task transport", () => {
  it("uses GET only when a known task is resumed", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(complete)); vi.stubGlobal("fetch", fetcher);
    await runRequest({ ...intent(), job_id: "job-1" }, () => {}, new AbortController().signal);
    expect(fetcher).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/jobs/job-1");
    expect(fetcher.mock.calls[0][1].method).toBeUndefined();
  });
  it("submits only business fields with the unchanged idempotency key", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(complete, 202)); vi.stubGlobal("fetch", fetcher);
    await runRequest(intent(), () => {}, new AbortController().signal);
    const options = fetcher.mock.calls[0][1];
    expect(options.headers["Idempotency-Key"]).toBe(intent().key);
    expect(JSON.parse(options.body)).toEqual({ query: "test", paper_ids: [], dataset_id: "api-test" });
  });
  it("preserves queue-full status without inventing a task result", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ message: "队列已满" }, 429)));
    await expect(runRequest(intent(), () => {}, new AbortController().signal)).rejects.toMatchObject({ status: 429, message: "队列已满" });
  });
  it("rejects results from a different dataset", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ ...complete, dataset_id: "another" })));
    await expect(runRequest(intent(), () => {}, new AbortController().signal)).rejects.toBeInstanceOf(ApiFailure);
  });
});
