"use client";

import { useState } from "react";
import { Notice } from "@/components/Notice";
import { PageHeader } from "@/components/PageHeader";
import { ProgressBar } from "@/components/ProgressBar";
import { StatCard } from "@/components/StatCard";
import { formatCurrency, getJson } from "@/lib/api";
import type { RiskStatus } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function RiskPage() {
  const { data, loading, error, refreshedAt, refresh } = usePolling<RiskStatus>("/api/risk/status");
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const capital = Number(data?.capital || 100000);
  const dailyLimit = Number(data?.daily_limit_usd || capital * Number(data?.daily_limit_pct || 0.02));
  const weeklyLimit = Number(data?.weekly_limit_usd || capital * Number(data?.weekly_limit_pct || 0.04));

  async function setKillSwitch(action: "arm" | "disarm") {
    setBusy(action);
    setActionError(null);
    try {
      await getJson("/api/risk/kill-switch", {
        method: "POST",
        body: JSON.stringify({ action, reason: action === "arm" ? "Manual dashboard trigger" : undefined })
      });
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHeader title="Risk Dashboard" subtitle="Persistent kill switch and loss controls" refreshedAt={refreshedAt} />
      <Notice loading={loading} error={error || actionError} />
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <StatCard label="Kill Switch" value={data?.kill_switch_armed ? "ARMED" : "DISARMED"} status={Boolean(data?.kill_switch_armed)} detail={data?.reason || "Portfolio-level control"} />
        <StatCard label="Daily Loss" value={formatCurrency(data?.daily_loss_usd)} status={Number(data?.daily_loss_usd || 0) >= dailyLimit * 0.8 ? "error" : "ok"} detail={`Limit ${formatCurrency(dailyLimit)}`} />
        <StatCard label="Weekly Loss" value={formatCurrency(data?.weekly_loss_usd)} status={Number(data?.weekly_loss_usd || 0) >= weeklyLimit * 0.8 ? "error" : "ok"} detail={`Limit ${formatCurrency(weeklyLimit)}`} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ProgressBar label="Daily loss meter" value={Number(data?.daily_loss_usd || 0)} limit={dailyLimit} />
        <ProgressBar label="Weekly loss meter" value={Number(data?.weekly_loss_usd || 0)} limit={weeklyLimit} />
      </div>
      <div className="mt-6 flex flex-wrap gap-3 rounded-lg border border-atlas-line bg-atlas-panel p-4">
        <button onClick={() => setKillSwitch("arm")} disabled={busy !== null} className="rounded-md bg-atlas-red px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-60">{busy === "arm" ? "Arming..." : "ARM"}</button>
        <button onClick={() => setKillSwitch("disarm")} disabled={busy !== null} className="rounded-md bg-atlas-green px-4 py-2 text-sm font-semibold text-atlas-bg transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-60">{busy === "disarm" ? "Disarming..." : "DISARM"}</button>
      </div>
    </>
  );
}
