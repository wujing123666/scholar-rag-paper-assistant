import { describe, expect, it } from "vitest";
import type { QuestionRecord } from "../api/contracts";
import { answerDemo, DATASET_ID, exampleQueries } from "../mocks/demo";
import { appendHistory, clearHistory, loadHistory, saveHistory } from "./history";
import { papers } from "../mocks/demo";
import { memoryStorage } from "../testStorage";

const record = (): QuestionRecord => ({
  id: "one", query: exampleQueries[0].query, paper_ids: ["demo_mesh"],
  created_at: "2026-10-15T00:00:00Z", dataset_id: DATASET_ID,
  result: answerDemo({ query: exampleQueries[0].query, paper_ids: [] }),
});

describe("shared API history", () => {
  const context = { datasetId: "api-history", sessionId: "owner", source: "api" as const, papers };
  const apiRecord = (id: string): QuestionRecord => ({ ...record(), id, dataset_id: context.datasetId,
    result: { ...record().result, source: "api", verification: { status: "verified" } } });
  it("appends independent answer keys and preserves legacy records", () => {
    const area = memoryStorage(); saveHistory([apiRecord("legacy")], area, context);
    appendHistory(apiRecord("first"), context, area); appendHistory(apiRecord("second"), context, area);
    expect(loadHistory(area, context).map(item => item.id).sort()).toEqual(["first", "legacy", "second"]);
    expect(area.getItem("scholarrag.api.history.v1:api-history:owner")).toBeNull();
  });
  it("caps persisted records without deleting unrelated datasets", () => {
    const area = memoryStorage(); area.setItem("unrelated", "keep");
    for (let index = 0; index < 25; index++) appendHistory(apiRecord(String(index)), context, area);
    expect(loadHistory(area, context)).toHaveLength(20); expect(area.length).toBe(21);
    clearHistory(context, area); expect(loadHistory(area, context)).toEqual([]); expect(area.getItem("unrelated")).toBe("keep");
  });
  it("ignores one damaged record without hiding intact answers", () => {
    const area = memoryStorage(); appendHistory(apiRecord("good"), context, area);
    area.setItem("scholarrag.api.history.v1:api-history:owner:record:bad", "{");
    expect(loadHistory(area, context).map(item => item.id)).toEqual(["good"]);
  });
});
const read = (value: unknown) => loadHistory({ getItem: () => JSON.stringify(value) });
describe("untrusted local demo history", () => {
  it("restores valid records and excludes other datasets", () => {
    expect(read([record(), { ...record(), dataset_id: "private-production" }])).toEqual([record()]);
  });
  it("recovers from invalid JSON and denied storage access", () => {
    expect(loadHistory({ getItem: () => "{" })).toEqual([]);
    expect(loadHistory({ getItem: () => { throw new Error("denied"); } })).toEqual([]);
  });
  it("discards malformed nested data and unknown citation references", () => {
    const invalid = record(); invalid.result.claims[0].citations = ["C99"];
    expect(read([invalid, { ...record(), result: { claims: [null], citations: [] } }])).toEqual([]);
  });
  it("discards non-existent files and invalid PDF pages", () => {
    const badFile = record(); badFile.result.citations[0].file_id = "other-file";
    const badPage = record(); badPage.result.citations[0].page_number = 3;
    expect(read([badFile, badPage])).toEqual([]);
  });
  it("does not restore API answers with a forged verified label into demo history", () => {
    const invalid = record(); invalid.result.source = "api"; invalid.result.verification.status = "verified";
    expect(read([invalid])).toEqual([]);
  });
  it("caps stored and restored records at twenty", () => {
    let serialized = "";
    expect(saveHistory(Array.from({ length: 25 }, (_, i) => ({ ...record(), id: String(i) })), {
      setItem: (_, value) => { serialized = value; },
    })).toBe(true);
    expect(JSON.parse(serialized)).toHaveLength(20);
    expect(read(Array.from({ length: 25 }, record))).toHaveLength(20);
  });
  it("reports quota errors to the caller", () => {
    expect(saveHistory([record()], { setItem: () => { throw new Error("quota"); } })).toBe(false);
  });
});
