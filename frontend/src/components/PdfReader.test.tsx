import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { papers, answerDemo, exampleQueries } from "../mocks/demo";
import PdfReader from "./PdfReader";

const pdf = vi.hoisted(() => ({ getDocument: vi.fn(), GlobalWorkerOptions: { workerSrc: "" } }));
vi.mock("pdfjs-dist", () => pdf);
const citation = () => answerDemo({ query: exampleQueries[0].query, paper_ids: [] }).citations[0];
function makeTask() {
  const renderPage = vi.fn(() => ({ promise: Promise.resolve(), cancel: vi.fn() }));
  const getPage = vi.fn(async () => ({
    getViewport: ({ scale }: { scale: number }) => ({ width: 600 * scale, height: 800 * scale }),
    render: renderPage,
  }));
  return { promise: Promise.resolve({ numPages: 2, getPage }), destroy: vi.fn(async () => {}), getPage, renderPage };
}
beforeEach(() => {
  pdf.getDocument.mockReset();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as CanvasRenderingContext2D);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("PDF file identity and resource lifecycle", () => {
  it("does not open another file when the cited file is missing", async () => {
    render(<PdfReader paper={papers[0]} citation={{ ...citation(), file_id: "missing" }} pageNumber={1} onPageChange={() => {}} />);
    await screen.findByText("引用对应的 PDF 文件不存在。");
    expect(pdf.getDocument).not.toHaveBeenCalled();
    expect(screen.queryByRole("link", { name: "下载原文" })).toBeNull();
  });
  it("rejects an invalid page and recovers when a valid page is selected", async () => {
    const task = makeTask(); pdf.getDocument.mockReturnValue(task);
    const { rerender } = render(<PdfReader paper={papers[0]} citation={citation()} pageNumber={3} onPageChange={() => {}} />);
    await screen.findByText("引用页码超出 PDF 范围，请下载原文核对。");
    expect(task.getPage).not.toHaveBeenCalled();
    rerender(<PdfReader paper={papers[0]} citation={citation()} pageNumber={2} onPageChange={() => {}} />);
    await waitFor(() => expect(task.getPage).toHaveBeenCalledWith(2));
    expect(screen.queryByRole("alert")).toBeNull();
  });
  it("cancels rendering and destroys the loading task on unmount", async () => {
    const task = makeTask(); pdf.getDocument.mockReturnValue(task);
    const { unmount } = render(<PdfReader paper={papers[0]} citation={citation()} pageNumber={1} onPageChange={() => {}} />);
    await waitFor(() => expect(task.renderPage).toHaveBeenCalled());
    const rendering = task.renderPage.mock.results[0].value;
    unmount();
    expect(rendering.cancel).toHaveBeenCalledOnce();
    expect(task.destroy).toHaveBeenCalledOnce();
  });
  it("shows a loading failure with a download fallback", async () => {
    pdf.getDocument.mockReturnValue({ promise: Promise.reject(new Error("PDF 文件不可读")), destroy: vi.fn(async () => {}) });
    render(<PdfReader paper={papers[0]} citation={citation()} pageNumber={1} onPageChange={() => {}} />);
    await screen.findByText("PDF 文件不可读");
    expect(screen.getByRole("link", { name: "下载原文" }).getAttribute("href")).toBe(papers[0].files[0].url);
  });
});
