"use client";

import { Notice } from "@/components/Notice";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { usePolling } from "@/lib/usePolling";
import type { HealthResponse } from "@/lib/types";

function serviceStatus(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && "status" in value) return String((value as { status?: unknown }).status || "unknown");
  if (value && typeof value === "object") return "ok";
  return "unknown";
}

function isErrorStatus(value: string): boolean {
  return value.toLowerCase().startsWith("error");
}

export default function OverviewPage() {
  const { data, loading, error, refreshedAt } = usePolling<HealthResponse>("/health");
  const services = data?.services || {};
  const dbStatus = serviceStatus(services.timescaledb);
  const killSwitch = serviceStatus(services.kill_switch);
  const registry = services.agent_registry as { running?: number; total?: number } | undefined;
  const strategies = services.strategies as { total?: number; active?: number } | undefined;

  return (
    <>
      <PageHeader title="System Overview" subtitle="Live ATLAS service health and operating state" refreshedAt={refreshedAt} />
      <Notice loading={loading} error={error} />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="TimescaleDB" value={isErrorStatus(dbStatus) ? "ERROR" : dbStatus.toUpperCase()} status={dbStatus} detail="GET /health" />
        <StatCard label="Kill Switch" value={killSwitch.toUpperCase()} status={killSwitch === "ARMED" || killSwitch === "armed"} detail="Persistent risk halt state" />
        <StatCard label="Total Strategies" value={strategies?.total ?? "-"} status="ok" detail={`${strategies?.active ?? 0} active`} />
        <StatCard label="Active Agents" value={registry?.running ?? "-"} status={(registry?.running ?? 0) > 0 ? "running" : "paused"} detail={`${registry?.total ?? 0} registered`} />
      </div>
      <section className="mt-6 rounded-lg border border-atlas-line bg-atlas-panel p-4">
        <h2 className="text-lg font-semibold">Service Payload</h2>
        <pre className="mt-4 max-h-[520px] overflow-auto rounded-md bg-atlas-bg p-4 text-xs text-atlas-muted">{JSON.stringify(data, null, 2)}</pre>
      </section>
    </>
  );
}
