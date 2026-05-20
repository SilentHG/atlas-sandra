import { statusTone } from "@/lib/api";

type StatCardProps = {
  label: string;
  value: string | number;
  detail?: string;
  status?: string | boolean;
};

export function StatCard({ label, value, detail, status }: StatCardProps) {
  return (
    <section className="rounded-lg border border-atlas-line bg-atlas-panel p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-atlas-muted">{label}</div>
      <div className={`mt-3 text-2xl font-semibold ${statusTone(status)}`}>{value}</div>
      {detail ? <div className="mt-2 min-h-5 text-sm text-atlas-muted">{detail}</div> : null}
    </section>
  );
}
