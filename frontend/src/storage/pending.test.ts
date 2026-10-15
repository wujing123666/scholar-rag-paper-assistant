import { describe, expect, it } from "vitest";
import { createPendingStore } from "./pending";
import type { PendingRequest } from "./pending";
import { memoryStorage } from "../testStorage";

const intent = (key = "checkpoint_test_0001"): PendingRequest => ({ key, job_id: "existing-job", query: "原问题",
  kind: "answer", paper_ids: [], dataset_id: "dataset", session_id: "session", created_at: Date.now() });
describe("per-tab task checkpoint", () => {
  it("keeps two tabs independent when one task completes", () => {
    const first = createPendingStore(memoryStorage(), null), second = createPendingStore(memoryStorage(), null);
    first.write(intent()); second.write(intent("checkpoint_test_0002")); first.clear(intent().key);
    expect(first.read()).toBeNull(); expect(second.read()?.key).toBe("checkpoint_test_0002");
  });
  it("does not clear a newer task with an older key", () => {
    const store = createPendingStore(memoryStorage(), null); store.write(intent("checkpoint_test_0002"));
    expect(store.clear(intent().key)).toBe(true); expect(store.read()?.key).toBe("checkpoint_test_0002");
  });
  it("preserves the job and idempotency key across a reload", () => {
    const area = memoryStorage(), original = intent(); createPendingStore(area, null).write(original);
    expect(createPendingStore(area, null).read()).toEqual(original);
  });
  it("migrates a legacy checkpoint without changing its job or key", () => {
    const area = memoryStorage(), legacy = memoryStorage(), original = intent();
    legacy.setItem("scholarrag.api.pending.v1", JSON.stringify(original));
    expect(createPendingStore(area, legacy).read()).toEqual(original);
    expect(legacy.getItem("scholarrag.api.pending.v1")).toBeNull();
    expect(createPendingStore(area, legacy).read()).toEqual(original);
  });
  it("retains the legacy checkpoint if migration cannot persist it", () => {
    const legacy = memoryStorage(); legacy.setItem("scholarrag.api.pending.v1", JSON.stringify(intent()));
    const blocked = { ...memoryStorage(), setItem: () => { throw new Error("quota"); } };
    expect(createPendingStore(blocked, legacy).read()).toBeNull();
    expect(legacy.getItem("scholarrag.api.pending.v1")).not.toBeNull();
  });
  it("handles denied storage without inventing a checkpoint", () => {
    const store = createPendingStore(null, null);
    expect(store.read()).toBeNull(); expect(store.write(intent())).toBe(false); expect(store.clear(intent().key)).toBe(false);
  });
});
