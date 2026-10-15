import { describe, expect, it, vi } from "vitest";
import { answerDemo, exampleQueries, papers, searchDemoPapers } from "./demo";

describe("explicit UI fixtures", () => {
  it.each(exampleQueries)("maps $paper_id to a real fixture file and page", ({ query, paper_id }) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const result = answerDemo({ query, paper_ids: [paper_id] });
    expect(result.source).toBe("demo");
    expect(result.verification.status).toBe("demo");
    expect(result.citations[0]).toMatchObject({ paper_id, file_id: paper_id, page_number: 1 });
    expect(papers.find(p => p.paper_id === paper_id)?.files[0].url).toContain(".pdf");
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });
  it("does not invent an answer for an unconfigured question about a known paper", () => {
    expect(() => answerDemo({ query: "DemoMesh是哪年发表的？", paper_ids: [] })).toThrow("尚未配置");
  });
  it("rejects an example outside the selected scope", () => {
    expect(() => answerDemo({ query: exampleQueries[0].query, paper_ids: ["demo_privacy"] })).toThrow("范围");
  });
  it("labels the preset refusal without claims or verification", () => {
    expect(answerDemo({ query: "DemoMesh的学习率是多少？", paper_ids: [] })).toMatchObject({
      status: "refused", claims: [], citations: [], verification: { status: "demo" }, source: "demo",
    });
  });
  it("returns no keyword fixture for an unrelated search", () => {
    expect(searchDemoPapers("量子计算容错门")).toEqual([]);
  });
});
