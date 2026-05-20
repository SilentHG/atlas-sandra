"use client";

import { useState } from "react";
import { EmptyRow } from "@/components/EmptyRow";
import { Notice } from "@/components/Notice";
import { PageHeader } from "@/components/PageHeader";
import { formatDate, getJson } from "@/lib/api";
import type { Strategy } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function StrategiesPage() {
  const { data, loading, error, refreshedAt, refresh } = usePolling<Strategy[]>("/api/strategies");
  const [actionError, setActionError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  async function generate() {
    setGenerating(true);
    setActionError(null);
    try {
      await getJson("/api/strategies/generate", {
        method: "POST",
        body: JSON.stringify({ strategy_type: "trend", symbols: ["AAPL", "MSFT"] })
      });
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  }

  return (
    <>
      <PageHeader title="Strategy Pipeline" subtitle="Generated strategies and lifecycle status" refreshedAt={refreshedAt} />
      <div className="mb-4 flex justify-end">
        <button onClick={generate} disabled={generating} className="rounded-md bg-atlas-blue px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60">
          {generating ? "Generating..." : "Generate New Strategy"}
        </button>
      </div>
      <Notice loading={loading} error={error || actionError} />
      <div className="overflow-hidden rounded-lg border border-atlas-line bg-atlas-panel">
        <table className="w-full table-fixed border-collapse">
          <thead className="bg-atlas-panel2 text-left text-xs uppercase tracking-wide text-atlas-muted">
            <tr><th className="px-4 py-3">Name</th><th className="px-4 py-3">Type</th><th className="px-4 py-3">Status</th><th className="px-4 py-3">Symbols</th><th className="px-4 py-3">Created Date</th></tr>
          </thead>
          <tbody className="divide-y divide-atlas-line text-sm">
            {(data || []).map((s) => (
              <tr key={s.id} className="hover:bg-atlas-panel2/60">
                <td className="px-4 py-3 font-medium">{s.name}</td>
                <td className="px-4 py-3 text-atlas-muted">{s.strategy_type}</td>
                <td className="px-4 py-3">{s.status}</td>
                <td className="px-4 py-3 text-atlas-muted">{Array.isArray(s.symbols) ? s.symbols.join(", ") : s.symbols}</td>
                <td className="px-4 py-3 text-atlas-muted">{formatDate(s.created_at)}</td>
              </tr>
            ))}
            {(data || []).length === 0 ? <EmptyRow colSpan={5} label="No strategies returned by the API." /> : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
