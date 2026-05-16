import { Activity, BriefcaseBusiness, ShieldAlert, TrendingUp } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

async function getJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return fallback;
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

type Portfolio = {
  open_count?: number;
  total_unrealized_pnl?: number;
  total_realized_pnl?: number;
  daily_pnl?: number;
};

type Position = {
  symbol: string;
  side: string;
  quantity: number;
  current_price?: number;
  unrealized_pnl?: number;
  pnl?: number;
};

type Strategy = {
  id: string;
  name: string;
  strategy_type: string;
  status: string;
  symbols: string[];
};

export default async function Page() {
  const [portfolio, positions, strategies, risk] = await Promise.all([
    getJson<Portfolio>("/portfolio", {}),
    getJson<Position[]>("/positions", []),
    getJson<Strategy[]>("/strategies", []),
    getJson<{ kill_switch_armed?: boolean; reason?: string }>("/api/risk/status", {})
  ]);

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>ATLAS</h1>
          <p>Shah Equity Holdings trading operations</p>
        </div>
        <div className={risk.kill_switch_armed ? "status danger" : "status ok"}>
          <ShieldAlert size={18} />
          {risk.kill_switch_armed ? "Trading halted" : "Risk gates armed"}
        </div>
      </header>

      <section className="metrics">
        <article>
          <BriefcaseBusiness size={22} />
          <span>Open Positions</span>
          <strong>{portfolio.open_count ?? positions.length}</strong>
        </article>
        <article>
          <TrendingUp size={22} />
          <span>Daily P&L</span>
          <strong>${Number(portfolio.daily_pnl ?? 0).toFixed(2)}</strong>
        </article>
        <article>
          <Activity size={22} />
          <span>Strategies</span>
          <strong>{strategies.length}</strong>
        </article>
      </section>

      <section className="grid">
        <div>
          <h2>Positions</h2>
          <table>
            <thead>
              <tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Mark</th><th>P&L</th></tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={`${p.symbol}-${p.side}`}>
                  <td>{p.symbol}</td>
                  <td>{p.side}</td>
                  <td>{Number(p.quantity).toFixed(4)}</td>
                  <td>${Number(p.current_price ?? 0).toFixed(2)}</td>
                  <td>${Number(p.unrealized_pnl ?? p.pnl ?? 0).toFixed(2)}</td>
                </tr>
              ))}
              {positions.length === 0 && <tr><td colSpan={5}>No open positions from the API.</td></tr>}
            </tbody>
          </table>
        </div>

        <div>
          <h2>Strategy Scoreboard</h2>
          <table>
            <thead>
              <tr><th>Name</th><th>Type</th><th>Status</th><th>Symbols</th></tr>
            </thead>
            <tbody>
              {strategies.map((s) => (
                <tr key={s.id}>
                  <td>{s.name}</td>
                  <td>{s.strategy_type}</td>
                  <td>{s.status}</td>
                  <td>{Array.isArray(s.symbols) ? s.symbols.join(", ") : ""}</td>
                </tr>
              ))}
              {strategies.length === 0 && <tr><td colSpan={4}>No strategies returned by the API.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
