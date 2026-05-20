"use client";

import { EmptyRow } from "@/components/EmptyRow";
import { Notice } from "@/components/Notice";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { formatCurrency, formatDate, formatNumber } from "@/lib/api";
import type { Portfolio } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function TradingPage() {
  const { data, loading, error, refreshedAt } = usePolling<Portfolio>("/api/portfolio/summary");
  const positions = data?.positions || [];

  return (
    <>
      <PageHeader title="Live Trading" subtitle="Portfolio P&L and open positions" refreshedAt={refreshedAt} />
      <Notice loading={loading} error={error} />
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <StatCard label="Daily P&L" value={formatCurrency(data?.daily_pnl)} status={Number(data?.daily_pnl || 0) >= 0 ? "ok" : "error"} detail="GET /api/portfolio/summary" />
        <StatCard label="Unrealized P&L" value={formatCurrency(data?.total_unrealized_pnl)} status={Number(data?.total_unrealized_pnl || 0) >= 0 ? "ok" : "error"} />
        <StatCard label="Open Positions" value={data?.open_count ?? positions.length} status="ok" />
      </div>
      <div className="overflow-hidden rounded-lg border border-atlas-line bg-atlas-panel">
        <table className="w-full table-fixed border-collapse">
          <thead className="bg-atlas-panel2 text-left text-xs uppercase tracking-wide text-atlas-muted">
            <tr><th className="px-4 py-3">Symbol</th><th className="px-4 py-3">Side</th><th className="px-4 py-3">Qty</th><th className="px-4 py-3">Entry</th><th className="px-4 py-3">Mark</th><th className="px-4 py-3">P&L</th><th className="px-4 py-3">Opened</th></tr>
          </thead>
          <tbody className="divide-y divide-atlas-line text-sm">
            {positions.map((p) => (
              <tr key={p.id || `${p.symbol}-${p.side}`} className="hover:bg-atlas-panel2/60">
                <td className="px-4 py-3 font-medium">{p.symbol}</td>
                <td className="px-4 py-3 text-atlas-muted">{p.side}</td>
                <td className="px-4 py-3">{formatNumber(p.quantity)}</td>
                <td className="px-4 py-3 text-atlas-muted">{formatCurrency(p.entry_price)}</td>
                <td className="px-4 py-3 text-atlas-muted">{formatCurrency(p.current_price)}</td>
                <td className="px-4 py-3">{formatCurrency(p.unrealized_pnl ?? p.pnl)}</td>
                <td className="px-4 py-3 text-atlas-muted">{formatDate(p.opened_at)}</td>
              </tr>
            ))}
            {positions.length === 0 ? <EmptyRow colSpan={7} label="No open positions returned by the API." /> : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
