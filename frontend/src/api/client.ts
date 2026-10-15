import type { AnswerJob, ApiHealth, Paper } from "./contracts";

export class ApiFailure extends Error {
  constructor(message: string, public status = 0) { super(message); }
}
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object";
async function json(path: string, init: RequestInit = {}): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...init, credentials: "same-origin",
      signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
    });
  } catch {
    throw new ApiFailure("无法连接服务。已有任务可能仍在运行，可刷新页面或恢复任务继续查看。");
  }
  let value: unknown;
  try { value = await response.json(); } catch { throw new ApiFailure("服务没有返回有效 JSON，请检查后端和代理配置。", response.status); }
  if (!response.ok) throw new ApiFailure(object(value) && typeof value.message === "string" ? value.message : "请求未能完成。", response.status);
  return value;
}
export async function loadBackend(): Promise<{ health: ApiHealth; papers: Paper[] }> {
  const health = await json("/health");
  if (!object(health) || health.ready !== true || typeof health.dataset_id !== "string" || typeof health.session_id !== "string")
    throw new ApiFailure("服务尚未就绪。");
  const library = await json("/papers");
  if (!object(library) || library.dataset_id !== health.dataset_id || !Array.isArray(library.papers) ||
      !library.papers.every(paper => object(paper) && typeof paper.paper_id === "string" && typeof paper.title === "string" &&
        Array.isArray(paper.files) && paper.files.every(file => object(file) && typeof file.file_id === "string" &&
          typeof file.url === "string" && file.url.startsWith("/api/v1/papers/"))))
    throw new ApiFailure("资料库接口格式不一致，请刷新后重试。");
  return { health: health as ApiHealth, papers: library.papers as Paper[] };
}

export type PendingRequest = {
  key: string; job_id: string | null; kind: "answer" | "search"; query: string; paper_ids: string[];
  dataset_id: string; session_id: string; created_at: number;
};
const PENDING_KEY = "scholarrag.api.pending.v1";
export function readPending(): PendingRequest | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
    if (!object(value) || typeof value.key !== "string" || !/^[a-zA-Z0-9_-]{16,80}$/.test(value.key) ||
      !(value.job_id === null || typeof value.job_id === "string") || !["answer", "search"].includes(String(value.kind)) ||
      typeof value.query !== "string" || value.query.length > 2000 || !Array.isArray(value.paper_ids) || value.paper_ids.length > 3 ||
      !value.paper_ids.every(id => typeof id === "string") || typeof value.dataset_id !== "string" ||
      typeof value.session_id !== "string" || typeof value.created_at !== "number") return null;
    return value as PendingRequest;
  } catch { return null; }
}
export function writePending(value: PendingRequest | null): boolean {
  try {
    if (value) localStorage.setItem(PENDING_KEY, JSON.stringify(value)); else localStorage.removeItem(PENDING_KEY);
    return true;
  } catch { return false; }
}
function job(value: unknown, intent: PendingRequest): AnswerJob {
  if (!object(value) || typeof value.job_id !== "string" || value.dataset_id !== intent.dataset_id || value.kind !== intent.kind ||
    !["queued", "running", "completed", "failed", "queue_timeout", "interrupted"].includes(String(value.state)))
    throw new ApiFailure("任务状态格式不一致。");
  if (value.state === "completed" && (!object(value.result) ||
    (intent.kind === "answer" ? value.result.source !== "api" || !Array.isArray(value.result.claims) || !Array.isArray(value.result.citations) : !Array.isArray(value.result.papers))))
    throw new ApiFailure("任务结果格式不一致。");
  return value as AnswerJob;
}
export async function runRequest(intent: PendingRequest, onProgress: (job: AnswerJob) => void, signal: AbortSignal): Promise<AnswerJob> {
  let current = intent.job_id ? job(await json(`/jobs/${intent.job_id}`, { signal }), intent) :
    job(await json(intent.kind === "answer" ? "/answers" : "/paper-searches", {
      method: "POST", signal,
      headers: { "Content-Type": "application/json", "Idempotency-Key": intent.key },
      body: JSON.stringify({ query: intent.query, paper_ids: intent.paper_ids, dataset_id: intent.dataset_id }),
    }), intent);
  onProgress(current);
  const deadline = Date.now() + 10 * 60_000;
  while (current.state === "queued" || current.state === "running") {
    if (Date.now() > deadline) throw new ApiFailure("等待结果超过十分钟，任务可能仍在运行。请恢复任务继续查看，避免重复提问。");
    await new Promise<void>((resolve, reject) => {
      if (signal.aborted) { reject(new Error("aborted")); return; }
      const aborted = () => { clearTimeout(timer); reject(new Error("aborted")); };
      const timer = setTimeout(() => { signal.removeEventListener("abort", aborted); resolve(); }, 1000);
      signal.addEventListener("abort", aborted, { once: true });
    });
    current = job(await json(`/jobs/${current.job_id}`, { signal }), intent);
    onProgress(current);
  }
  return current;
}
