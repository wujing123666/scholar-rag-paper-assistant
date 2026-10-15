export type PendingRequest = {
  key: string; job_id: string | null; kind: "answer" | "search"; query: string; paper_ids: string[];
  dataset_id: string; session_id: string; created_at: number;
};
const KEY = "scholarrag.api.pending.v2";
const LEGACY_KEY = "scholarrag.api.pending.v1";

function browserStorage(name: "sessionStorage" | "localStorage"): Storage | null {
  try { return window[name]; } catch { return null; }
}
function parse(raw: string | null): PendingRequest | null {
  try {
    const value = JSON.parse(raw || "null");
    if (!value || typeof value !== "object" || typeof value.key !== "string" || !/^[a-zA-Z0-9_-]{16,80}$/.test(value.key) ||
      !(value.job_id === null || typeof value.job_id === "string") || !["answer", "search"].includes(value.kind) ||
      typeof value.query !== "string" || !value.query.trim() || value.query.length > 2000 ||
      !Array.isArray(value.paper_ids) || value.paper_ids.length > 3 || !value.paper_ids.every((id: unknown) => typeof id === "string") ||
      typeof value.dataset_id !== "string" || typeof value.session_id !== "string" ||
      typeof value.created_at !== "number" || !Number.isFinite(value.created_at)) return null;
    return value;
  } catch { return null; }
}

/** Capture this tab's storage once; another tab cannot clear its checkpoint. */
export function createPendingStore(storage: Storage | null = browserStorage("sessionStorage"),
                                   legacy: Storage | null = browserStorage("localStorage")) {
  return {
    read(): PendingRequest | null {
      try {
        if (!storage) return null;
        const current = storage.getItem(KEY);
        if (current !== null) return parse(current);
        const previous = parse(legacy?.getItem(LEGACY_KEY) ?? null);
        if (!previous) return null;
        // Keep the old job/key, and only remove the legacy copy after a durable write.
        storage.setItem(KEY, JSON.stringify(previous));
        if (parse(legacy?.getItem(LEGACY_KEY) ?? null)?.key === previous.key) legacy?.removeItem(LEGACY_KEY);
        return previous;
      } catch { return null; }
    },
    write(value: PendingRequest): boolean {
      try { if (!storage) return false; storage.setItem(KEY, JSON.stringify(value)); return true; }
      catch { return false; }
    },
    clear(expectedKey: string): boolean {
      try {
        if (!storage) return false;
        if (parse(storage.getItem(KEY))?.key === expectedKey) storage.removeItem(KEY);
        return true;
      } catch { return false; }
    },
  };
}
export function readPending(storage?: Storage): PendingRequest | null { return createPendingStore(storage).read(); }
export function writePending(value: PendingRequest, storage?: Storage): boolean { return createPendingStore(storage).write(value); }
export function clearPending(key: string, storage?: Storage): boolean { return createPendingStore(storage).clear(key); }
