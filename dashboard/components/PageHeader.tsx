type PageHeaderProps = {
  title: string;
  subtitle: string;
  refreshedAt?: string | null;
};

export function PageHeader({ title, subtitle, refreshedAt }: PageHeaderProps) {
  return (
    <div className="mb-6 flex flex-col gap-2 border-b border-atlas-line pb-5 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-wide text-atlas-text">{title}</h1>
        <p className="mt-1 text-sm text-atlas-muted">{subtitle}</p>
      </div>
      <div className="text-xs text-atlas-muted">
        Refreshes every 10s{refreshedAt ? ` | Last ${new Date(refreshedAt).toLocaleTimeString()}` : ""}
      </div>
    </div>
  );
}
