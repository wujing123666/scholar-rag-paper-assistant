import type { Paper, QuestionRecord } from "../api/contracts";
import { DATASET_ID, papers } from "../mocks/demo";
const KEY = "scholarrag.demo.history.v1";
const LIMIT = 20;
export type HistoryContext = { datasetId: string; papers: Paper[]; source: "demo" | "api"; sessionId?: string };
const DEMO_CONTEXT: HistoryContext = { datasetId: DATASET_ID, papers, source: "demo" };
const keyFor = (context: HistoryContext) => context.source === "demo" ? KEY :
  `scholarrag.api.history.v1:${context.datasetId}:${context.sessionId}`;
type HistoryReader = Pick<Storage, "getItem"> & Partial<Pick<Storage, "length" | "key">>;
export function getHistoryStorage(): Storage | null {
  try { return window.localStorage; } catch { return null; }
}
const recordKey = (context: HistoryContext, id: string) => `${keyFor(context)}:record:${encodeURIComponent(id)}`;
function recordKeys(storage: HistoryReader, context: HistoryContext): string[] {
  if (typeof storage.length !== "number" || !storage.key) return [];
  const keys: string[] = [];
  for (let index = 0; index < storage.length; index++) {
    const key = storage.key(index);
    if (key?.startsWith(`${keyFor(context)}:record:`)) keys.push(key);
  }
  return keys;
}
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(v => typeof v === "string");

function validRecord(item: unknown, context: HistoryContext): item is QuestionRecord {
  if (!object(item) || item.dataset_id !== context.datasetId || typeof item.id !== "string" ||
    typeof item.query !== "string" || !strings(item.paper_ids) ||
    !item.paper_ids.every(id => context.papers.some(p => p.paper_id === id)) ||
    typeof item.created_at !== "string" || !Number.isFinite(Date.parse(item.created_at))) return false;
  const result = item.result;
  if (!object(result) || result.source !== context.source || !object(result.verification) ||
    !(context.source === "demo" ? result.verification.status === "demo" :
      ["verified", "skipped", "unavailable"].includes(String(result.verification.status))) ||
    !["answered", "refused", "evidence_only"].includes(String(result.status)) ||
    !(result.reason === null || typeof result.reason === "string") ||
    !Array.isArray(result.claims) || !Array.isArray(result.citations)) return false;
  const ids = new Set<string>();
  for (const citation of result.citations) {
    if (!object(citation) || !["citation_id", "paper_id", "file_id", "paper_title", "section", "chunk_id", "excerpt"]
      .every(key => typeof citation[key] === "string") || typeof citation.page_number !== "number" ||
      !Number.isInteger(citation.page_number) || citation.page_number < 1 ||
      (context.source === "demo" && citation.page_number > 2)) return false;
    const paper = context.papers.find(p => p.paper_id === citation.paper_id);
    if (!paper?.files.some(file => file.file_id === citation.file_id) || ids.has(citation.citation_id as string)) return false;
    ids.add(citation.citation_id as string);
  }
  return result.claims.every(claim => object(claim) && typeof claim.text === "string" &&
    strings(claim.citations) && claim.citations.every(id => ids.has(id)));
}

export function loadHistory(storage: HistoryReader | null = getHistoryStorage(), context: HistoryContext = DEMO_CONTEXT): QuestionRecord[] {
  try {
    if (!storage) return [];
    let parsed: unknown = [];
    try { parsed = JSON.parse(storage.getItem(keyFor(context)) || "[]"); } catch { /* Legacy snapshot may be damaged. */ }
    const legacy = Array.isArray(parsed) ? parsed.filter(item => validRecord(item, context)).slice(0, LIMIT) : [];
    if (context.source === "demo") return legacy;
    const records = new Map<string, QuestionRecord>(legacy.map(item => [item.id, item]));
    for (const key of recordKeys(storage, context)) {
      try {
        const item: unknown = JSON.parse(storage.getItem(key) || "null");
        if (validRecord(item, context) && key === recordKey(context, item.id)) records.set(item.id, item);
      } catch { /* Ignore a damaged record without losing unrelated answers. */ }
    }
    return [...records.values()].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.id.localeCompare(a.id)).slice(0, LIMIT);
  } catch { return []; }
}
export function saveHistory(records: QuestionRecord[], storage: Pick<Storage, "setItem"> | null = getHistoryStorage(), context: HistoryContext = DEMO_CONTEXT): boolean {
  try {
    if (!storage) return false;
    storage.setItem(keyFor(context), JSON.stringify(records.filter(item => validRecord(item, context)).slice(0, LIMIT)));
    return true;
  } catch { return false; }
}

/** Each API answer owns one key; stale tab snapshots cannot replace another answer. */
export function appendHistory(record: QuestionRecord, context: HistoryContext, storage: Storage | null = getHistoryStorage()): boolean {
  try {
    if (!storage || !validRecord(record, context)) return false;
    if (context.source === "demo") return saveHistory([record, ...loadHistory(storage, context)], storage, context);
    const existing = loadHistory({ getItem: key => storage.getItem(key) }, context);
    storage.setItem(recordKey(context, record.id), JSON.stringify(record));
    for (const previous of existing) {
      const key = recordKey(context, previous.id);
      if (storage.getItem(key) === null) storage.setItem(key, JSON.stringify(previous));
    }
    storage.removeItem(keyFor(context));
    const candidates = recordKeys(storage, context);
    const keep = new Set(loadHistory(storage, context).map(item => recordKey(context, item.id)));
    for (const key of candidates) if (!keep.has(key)) storage.removeItem(key);
    return true;
  } catch { return false; }
}
export function clearHistory(context: HistoryContext, storage: Storage | null = getHistoryStorage()): boolean {
  try {
    if (!storage) return false;
    storage.removeItem(keyFor(context));
    for (const key of recordKeys(storage, context)) storage.removeItem(key);
    return true;
  } catch { return false; }
}
