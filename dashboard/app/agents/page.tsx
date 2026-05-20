"use client";

import { EmptyRow } from "@/components/EmptyRow";
import { Notice } from "@/components/Notice";
import { PageHeader } from "@/components/PageHeader";
import { formatDate } from "@/lib/api";
import type { Agent } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function AgentsPage() {
  const { data, loading, error, refreshedAt } = usePolling<Agent[]>("/api/agents/registry");
  const agents = data || [];

  return (
    <>
      <PageHeader title="Agent Registry" subtitle="Registered ATLAS agents and heartbeat freshness" refreshedAt={refreshedAt} />
      <Notice loading={loading} error={error} />
      <div className="overflow-hidden rounded-lg border border-atlas-line bg-atlas-panel">
        <table className="w-full table-fixed border-collapse">
          <thead className="bg-atlas-panel2 text-left text-xs uppercase tracking-wide text-atlas-muted">
            <tr><th className="px-4 py-3">Name</th><th className="px-4 py-3">Status</th><th className="px-4 py-3">Last Heartbeat</th></tr>
          </thead>
          <tbody className="divide-y divide-atlas-line text-sm">
            {agents.map((agent) => (
              <tr key={agent.id || agent.name} className="hover:bg-atlas-panel2/60">
                <td className="px-4 py-3 font-medium">{agent.name}</td>
                <td className="px-4 py-3 text-atlas-muted">{agent.status}</td>
                <td className="px-4 py-3 text-atlas-muted">{formatDate(agent.last_heartbeat)}</td>
              </tr>
            ))}
            {agents.length === 0 ? <EmptyRow colSpan={3} label="No agents returned by the API." /> : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
