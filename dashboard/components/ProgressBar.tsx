type ProgressBarProps = {
  label: string;
  value: number;
  limit: number;
};

export function ProgressBar({ label, value, limit }: ProgressBarProps) {
  const pct = limit > 0 ? Math.min(100, Math.max(0, (value / limit) * 100)) : 0;
  const tone = pct >= 80 ? "bg-atlas-red" : pct >= 60 ? "bg-atlas-amber" : "bg-atlas-green";
  return (
    <div className="rounded-lg border border-atlas-line bg-atlas-panel p-4">
      <div className="flex items-center justify-between gap-4 text-sm">
        <span className="font-medium text-atlas-text">{label}</span>
        <span className="text-atlas-muted">{pct.toFixed(1)}%</span>
      </div>
      <div className="mt-3 h-3 overflow-hidden rounded-full bg-atlas-panel2">
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 text-xs text-atlas-muted">${value.toFixed(2)} / ${limit.toFixed(2)}</div>
    </div>
  );
}
