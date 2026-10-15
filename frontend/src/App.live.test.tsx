import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnswerJob, AnswerResult } from "./api/contracts";
import { papers } from "./mocks/demo";
import { ApiFailure, readPending } from "./api/client";
import App from "./App";

const api = vi.hoisted(() => ({ run: vi.fn() }));
const data = {
  health: { ready: true, dataset_id: "api-test", session_id: "browser-test", papers_count: 3, label: "测试论文库", verification_mode: "single" },
  papers,
};
vi.mock("./api/useBackend", () => ({ useBackend: () => ({ data, error: "", retry: vi.fn() }) }));
vi.mock("./api/client", async original => ({ ...await original<typeof import("./api/client")>(), runRequest: api.run }));
vi.mock("./components/PdfReader", () => ({ default: () => <aside aria-label="原文阅读区" /> }));
const answer: AnswerResult = {
  status: "answered", source: "api", reason: null, verification: { status: "verified" },
  claims: [{ text: "来自真实接口的结论", citations: [] }], citations: [],
};
const completed = (): AnswerJob => ({ job_id: "same-job", dataset_id: "api-test", kind: "answer", state: "completed", result: answer, error: null });
beforeEach(() => {
  window.history.replaceState({}, "", "/"); localStorage.clear(); api.run.mockReset();
});
afterEach(cleanup);
function submit() {
  fireEvent.change(screen.getByRole("textbox", { name: "输入你的论文问题" }), { target: { value: "这个不是预设示例的问题" } });
  fireEvent.click(screen.getByRole("button", { name: "提交问题" }));
}

describe("live RAG workspace", () => {
  it("submits arbitrary questions to the real adapter without a demo fallback", async () => {
    api.run.mockResolvedValue(completed()); render(<App />); submit();
    await screen.findByText("来自真实接口的结论");
    expect(api.run.mock.calls[0][0]).toMatchObject({ query: "这个不是预设示例的问题", dataset_id: "api-test", kind: "answer" });
    expect(screen.queryByText(/尚未配置界面示例/)).toBeNull();
    expect(readPending()).toBeNull();
  });
  it("keeps the server task alive while browsing and saves its result without taking over", async () => {
    let finish!: (job: AnswerJob) => void;
    api.run.mockImplementation(() => new Promise<AnswerJob>(resolve => { finish = resolve; }));
    render(<App />); submit();
    fireEvent.click(screen.getByRole("button", { name: /新建问题/ }));
    await act(async () => { finish(completed()); });
    await waitFor(() => expect(screen.getByRole("button", { name: "这个不是预设示例的问题" })).toBeTruthy());
    expect(screen.queryByText("来自真实接口的结论")).toBeNull();
    expect(api.run).toHaveBeenCalledOnce();
  });
  it("restores the saved job ID after reload without making a new task", async () => {
    localStorage.setItem("scholarrag.api.pending.v1", JSON.stringify({
      key: "idempotency_saved_01", job_id: "existing-job", kind: "answer", query: "刷新前的问题", paper_ids: [],
      dataset_id: "api-test", session_id: "browser-test", created_at: Date.now(),
    }));
    api.run.mockResolvedValue(completed()); render(<App />);
    await screen.findByText("来自真实接口的结论");
    expect(api.run.mock.calls[0][0].job_id).toBe("existing-job");
  });
  it("keeps the same idempotency key when recovering an uncertain network result", async () => {
    api.run.mockRejectedValueOnce(new ApiFailure("网络断开")); render(<App />); submit();
    await screen.findByText("网络断开");
    const key = readPending()?.key;
    api.run.mockResolvedValueOnce(completed());
    fireEvent.click(screen.getByRole("button", { name: "恢复任务" }));
    await screen.findByText("来自真实接口的结论");
    expect(api.run.mock.calls[1][0].key).toBe(key);
  });
  it("shows a server failure rather than substituting a successful demo answer", async () => {
    api.run.mockResolvedValue({ ...completed(), state: "failed", result: null,
      error: { code: "TASK_FAILED", message: "真实服务失败", request_id: "same-job" } });
    render(<App />); submit();
    await screen.findByText("真实服务失败");
    expect(screen.queryByText("演示答案")).toBeNull();
  });
});
