import { StrictMode } from "react";
import { act, cleanup, fireEvent, render, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnswerJob, AnswerResult } from "./api/contracts";
import type { PendingRequest } from "./api/client";
import { ApiFailure, readPending } from "./api/client";
import { papers as demoPapers } from "./mocks/demo";
import { loadHistory } from "./storage/history";
import { memoryStorage } from "./testStorage";
import App from "./App";

const transport = vi.hoisted(() => ({ run: vi.fn() }));
vi.mock("./api/client", async original => ({ ...await original<typeof import("./api/client")>(), runRequest: transport.run }));
vi.mock("./components/PdfReader", () => ({ default: () => <aside aria-label="原文阅读区" /> }));
const papers = demoPapers.map(paper => ({ ...paper, files: paper.files.map(file => ({ ...file,
  url: `/api/v1/papers/${paper.paper_id}/files/${file.file_id}` })) }));
const health = { ready: true, dataset_id: "session-test", session_id: "cookie-owner", papers_count: 3,
  label: "隔离论文库", verification_mode: "single" };
const historyContext = { datasetId: health.dataset_id, sessionId: health.session_id, source: "api" as const, papers };
const result: AnswerResult = { status: "answered", source: "api", reason: null,
  verification: { status: "verified" }, claims: [{ text: "隔离答案", citations: [] }], citations: [] };
const response = (value: unknown) => ({ ok: true, status: 200, json: async () => value });
const finishers = new Map<string, (value: AnswerJob) => void>();
let fetcher: ReturnType<typeof vi.fn>;
beforeEach(() => {
  window.history.replaceState({}, "", "/"); localStorage.clear(); sessionStorage.clear();
  finishers.clear(); transport.run.mockReset();
  fetcher = vi.fn(async (path: string) => response(path.endsWith("/health") ? health : { dataset_id: health.dataset_id, papers }));
  vi.stubGlobal("fetch", fetcher);
  transport.run.mockImplementation((intent: PendingRequest, progress: (job: AnswerJob) => void, signal: AbortSignal) => {
    progress({ job_id: `job-${intent.query}`, dataset_id: health.dataset_id, kind: "answer", state: "running", result: null, error: null });
    return new Promise<AnswerJob>((resolve, reject) => {
      finishers.set(intent.query, resolve);
      signal.addEventListener("abort", () => reject(new ApiFailure("disposed")), { once: true });
    });
  });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
async function ready(container: HTMLElement) {
  await waitFor(() => expect(within(container).getByRole("button", { name: /^论文库\s*3$/ })).toBeTruthy());
}
function submit(container: HTMLElement, query: string) {
  fireEvent.change(within(container).getByRole("textbox"), { target: { value: query } });
  fireEvent.click(within(container).getByRole("button", { name: "提交问题" }));
}
async function finish(query: string) {
  await act(async () => finishers.get(query)!({ job_id: `job-${query}`, dataset_id: health.dataset_id,
    kind: "answer", state: "completed", result, error: null }));
}
describe("browser session lifecycle regressions", () => {
  it("one tab finishing leaves another tab's checkpoint recoverable, and keeps both answers", async () => {
    const areaA = memoryStorage(), areaB = memoryStorage();
    vi.stubGlobal("sessionStorage", areaA); const first = render(<App />);
    vi.stubGlobal("sessionStorage", areaB); const second = render(<App />);
    await ready(first.container); await ready(second.container);
    submit(first.container, "A"); submit(second.container, "B");
    const originalB = readPending(areaB);
    expect(readPending(areaA)?.query).toBe("A"); expect(originalB?.query).toBe("B");
    await finish("A"); expect(readPending(areaA)).toBeNull(); expect(readPending(areaB)).toEqual(originalB);
    second.unmount(); const refreshed = render(<App />); await ready(refreshed.container);
    await waitFor(() => expect(transport.run).toHaveBeenCalledTimes(3));
    expect(transport.run.mock.calls[2][0]).toMatchObject({ key: originalB!.key, job_id: originalB!.job_id });
    await finish("B");
    expect(loadHistory(undefined, historyContext).map(item => item.query).sort()).toEqual(["A", "B"]);
    fireEvent(window, new StorageEvent("storage", { key: "shared-history-change" }));
    expect(within(first.container).getByRole("button", { name: "B" })).toBeTruthy();
  });
  it("clearing history does not let a background tab resurrect its stale snapshot", async () => {
    vi.stubGlobal("sessionStorage", memoryStorage()); const first = render(<App />);
    vi.stubGlobal("sessionStorage", memoryStorage()); const second = render(<App />);
    await ready(first.container); await ready(second.container);
    submit(first.container, "A"); submit(second.container, "B"); await finish("A");
    fireEvent.click(within(first.container).getByRole("button", { name: "清空本地记录" })); await finish("B");
    expect(loadHistory(undefined, historyContext).map(item => item.query)).toEqual(["B"]);
  });
  it("StrictMode shares one initial health request and preserves recovery after reload", async () => {
    let release!: () => void;
    let calls = 0;
    fetcher.mockImplementation((path: string) => {
      if (!path.endsWith("/health")) return Promise.resolve(response({ dataset_id: health.dataset_id, papers }));
      calls++;
      if (calls === 1) return new Promise(resolve => { release = () => resolve(response(health)); });
      return Promise.resolve(response(health));
    });
    const first = render(<StrictMode><App /></StrictMode>);
    await act(async () => {}); await act(async () => release()); await ready(first.container);
    expect(calls).toBe(1); submit(first.container, "恢复问题"); const original = readPending(); first.unmount();
    const refreshed = render(<App />); await ready(refreshed.container);
    await waitFor(() => expect(transport.run).toHaveBeenCalledTimes(2));
    expect(transport.run.mock.calls[1][0]).toMatchObject({ key: original!.key, job_id: original!.job_id, session_id: health.session_id });
    expect(within(refreshed.container).queryByText(/之前的任务属于其他资料库、会话/)).toBeNull();
    await finish("恢复问题");
  });
  it("submits on HTTP without randomUUID", async () => {
    const tab = render(<App />); await ready(tab.container);
    vi.stubGlobal("crypto", { getRandomValues: crypto.getRandomValues.bind(crypto) }); submit(tab.container, "HTTP问题");
    expect(transport.run).toHaveBeenCalledOnce(); expect(readPending()?.key).toMatch(/^[a-f0-9]{32}$/);
    await finish("HTTP问题"); expect(readPending()).toBeNull();
    submit(tab.container, "HTTP第二题"); expect(transport.run).toHaveBeenCalledTimes(2); await finish("HTTP第二题");
  });
  it("a secure-random failure does not leave the UI busy and can be retried", async () => {
    const tab = render(<App />); await ready(tab.container); const originalCrypto = crypto;
    vi.stubGlobal("crypto", { getRandomValues: () => { throw new Error("unavailable"); } }); submit(tab.container, "重试问题");
    expect(within(tab.container).getByText(/无法创建安全请求编号/)).toBeTruthy();
    expect(within(tab.container).getByRole("button", { name: "提交问题" }).hasAttribute("disabled")).toBe(false);
    expect(transport.run).not.toHaveBeenCalled();
    vi.stubGlobal("crypto", originalCrypto); submit(tab.container, "重试问题");
    expect(transport.run).toHaveBeenCalledOnce(); await finish("重试问题");
  });
});
