type NoticeProps = {
  error?: string | null;
  loading?: boolean;
};

export function Notice({ error, loading }: NoticeProps) {
  if (error) {
    return <div className="mb-4 rounded-md border border-atlas-red/40 bg-atlas-red/10 px-3 py-2 text-sm text-atlas-red">{error}</div>;
  }
  if (loading) {
    return <div className="mb-4 rounded-md border border-atlas-line bg-atlas-panel px-3 py-2 text-sm text-atlas-muted">Loading live API data...</div>;
  }
  return null;
}
