export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

export type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  refreshedAt: string | null;
};

export async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    },
    cache: "no-store"
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ""}`);
  }

  return (await res.json()) as T;
}

export function formatCurrency(value: unknown): string {
  const n = Number(value ?? 0);
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(Number.isFinite(n) ? n : 0);
}

export function formatDate(value: unknown): string {
  if (!value) return "-";
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

export function formatNumber(value: unknown): string {
  const n = Number(value ?? 0);
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(Number.isFinite(n) ? n : 0);
}

export function statusTone(status?: string | boolean): string {
  if (typeof status === "boolean") return status ? "text-atlas-red" : "text-atlas-green";
  const normalized = String(status || "").toLowerCase();
  if (["ok", "healthy", "running", "active", "disarmed"].includes(normalized)) return "text-atlas-green";
  if (["stale", "paused", "pending", "degraded"].includes(normalized)) return "text-atlas-amber";
  if (["error", "failed", "armed", "stopped", "halted"].includes(normalized)) return "text-atlas-red";
  return "text-atlas-muted";
}
