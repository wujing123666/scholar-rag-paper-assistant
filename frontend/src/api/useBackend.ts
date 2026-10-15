import { useEffect, useState } from "react";
import { loadBackend } from "./client";
import type { ApiHealth, Paper } from "./contracts";

export function useBackend(demoMode: boolean) {
  const [data, setData] = useState<{ health: ApiHealth; papers: Paper[] } | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (demoMode) return;
    let disposed = false;
    setError(""); setData(null);
    loadBackend().then(value => { if (!disposed) setData(value); })
      .catch(reason => { if (!disposed) setError(reason instanceof Error ? reason.message : "连接服务失败。"); });
    return () => { disposed = true; };
  }, [demoMode, attempt]);
  return { data, error, retry: () => setAttempt(value => value + 1) };
}
