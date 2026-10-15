import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { answerDemo, DATASET_ID, exampleQueries } from "./mocks/demo";

vi.mock("./components/PdfReader", () => ({ default: ({ paper }: { paper: { paper_id: string } | null }) =>
  <aside aria-label="原文阅读区" data-paper-id={paper?.paper_id || ""} /> }));
beforeEach(() => window.history.replaceState({}, "", "/?demo=1"));
afterEach(() => { cleanup(); localStorage.clear(); vi.unstubAllGlobals(); });
describe("workspace request lifecycle", () => {
  it("keeps a submitted scope when the next question scope changes", async () => {
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "输入你的论文问题" }), { target: { value: "DemoMesh怎样补全缺失读数？" } });
    fireEvent.click(screen.getByRole("button", { name: "提交问题" }));
    await screen.findByText("演示答案");
    fireEvent.click(screen.getByRole("button", { name: "全部示例论文" }));
    fireEvent.click(screen.getByRole("button", { name: "DemoShield" }));
    expect(document.querySelector(".question-heading")?.textContent).toContain("全部示例论文");
    expect(document.querySelector(".scope-button")?.textContent).toContain("DemoShield");
  });
  it("does not publish a late result after starting a new question", async () => {
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "输入你的论文问题" }), { target: { value: "DemoMesh怎样补全缺失读数？" } });
    fireEvent.click(screen.getByRole("button", { name: "提交问题" }));
    fireEvent.click(screen.getByRole("button", { name: /新建问题/ }));
    await waitFor(() => expect(screen.queryByText("正在载入界面示例…")).toBeNull());
    // Wait past the fixture delay to verify stale completion is actually ignored.
    await new Promise(resolve => setTimeout(resolve, 180));
    expect(screen.queryByText("演示答案")).toBeNull();
    expect(JSON.parse(localStorage.getItem("scholarrag.demo.history.v1") || "[]")).toEqual([]);
  });
  it("keeps actual upload disabled", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "论文入库" }));
    expect((screen.getByRole("button", { name: "上传功能尚未开放" }) as HTMLButtonElement).disabled).toBe(true);
  });
  it("does not let a pending answer replace a paper opened from the library", async () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false }));
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "输入你的论文问题" }), { target: { value: "DemoMesh怎样补全缺失读数？" } });
    fireEvent.click(screen.getByRole("button", { name: "提交问题" }));
    fireEvent.click(screen.getByRole("button", { name: /论文库\s*3/ }));
    fireEvent.change(screen.getByRole("textbox", { name: "筛选论文" }), { target: { value: "隐私" } });
    fireEvent.click(screen.getByRole("button", { name: "阅读原文" }));
    await new Promise(resolve => setTimeout(resolve, 180));
    expect(screen.getByRole("complementary", { name: "原文阅读区" }).getAttribute("data-paper-id")).toBe("demo_privacy");
    expect(screen.queryByText("演示答案")).toBeNull();
  });
  it("keeps identical C1 labels isolated when opening different historical answers", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false }));
    localStorage.setItem("scholarrag.demo.history.v1", JSON.stringify(exampleQueries.slice(0, 2).map((example, i) => ({
      id: String(i), query: example.query, paper_ids: [example.paper_id], dataset_id: DATASET_ID,
      created_at: "2026-10-15T00:00:00Z", result: answerDemo({ query: example.query, paper_ids: [] }),
    }))));
    render(<App />);
    for (const example of exampleQueries.slice(0, 2)) {
      fireEvent.click(screen.getByRole("button", { name: example.query }));
      fireEvent.click(screen.getAllByRole("button", { name: "查看引用 C1，第1页" })[0]);
      expect(screen.getByRole("complementary", { name: "原文阅读区" }).getAttribute("data-paper-id")).toBe(example.paper_id);
      expect(screen.getByRole("heading", { name: example.query })).toBeTruthy();
    }
  });
  it("clears only the local history in the isolated test browser", async () => {
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "输入你的论文问题" }), { target: { value: exampleQueries[0].query } });
    fireEvent.click(screen.getByRole("button", { name: "提交问题" }));
    await screen.findByText("演示答案");
    fireEvent.click(screen.getByRole("button", { name: "清空示例记录" }));
    expect(JSON.parse(localStorage.getItem("scholarrag.demo.history.v1") || "[]")).toEqual([]);
    expect(screen.queryByRole("button", { name: "清空示例记录" })).toBeNull();
  });
  it("returns a newly opened question to the top without focus scrolling it away", async () => {
    render(<App />);
    const workspace = document.querySelector(".workspace")!;
    const conversation = document.querySelector(".conversation-scroll")!;
    workspace.scrollTop = 499; conversation.scrollTop = 120;
    const focus = vi.spyOn(HTMLTextAreaElement.prototype, "focus");
    fireEvent.click(screen.getByRole("button", { name: /新建问题/ }));
    await waitFor(() => expect(workspace.scrollTop).toBe(0));
    expect(conversation.scrollTop).toBe(0);
    expect(focus).toHaveBeenCalledWith({ preventScroll: true });
    focus.mockRestore();
  });
});
