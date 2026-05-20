"use client";

import { useCallback, useEffect, useState } from "react";
import { getJson, type ApiState } from "@/lib/api";

export function usePolling<T>(path: string, intervalMs = 10000): ApiState<T> & { refresh: () => Promise<void> } {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
    refreshedAt: null
  });

  const refresh = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true }));
    try {
      const data = await getJson<T>(path);
      setState({ data, loading: false, error: null, refreshedAt: new Date().toISOString() });
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : String(error),
        refreshedAt: new Date().toISOString()
      }));
    }
  }, [path]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (!cancelled) await refresh();
    };
    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs, refresh]);

  return { ...state, refresh };
}
