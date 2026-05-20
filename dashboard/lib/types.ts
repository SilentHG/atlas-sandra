export type HealthResponse = {
  status?: string;
  services?: Record<string, any>;
  timestamp?: string;
};

export type Strategy = {
  id: string;
  name: string;
  strategy_type: string;
  status: string;
  symbols: string[] | string;
  created_at?: string;
};

export type Portfolio = {
  positions?: Position[];
  open_count?: number;
  total_unrealized_pnl?: number;
  total_realized_pnl?: number;
  daily_pnl?: number;
};

export type Position = {
  id?: string;
  symbol: string;
  side: string;
  quantity: number | string;
  entry_price?: number | string;
  current_price?: number | string;
  unrealized_pnl?: number | string;
  realized_pnl?: number | string;
  pnl?: number | string;
  status?: string;
  opened_at?: string;
};

export type RiskStatus = {
  kill_switch_armed?: boolean;
  reason?: string | null;
  armed_at?: string;
  daily_loss_usd?: number;
  weekly_loss_usd?: number;
  capital?: number;
  daily_limit_pct?: number;
  weekly_limit_pct?: number;
  daily_limit_usd?: number;
  weekly_limit_usd?: number;
};

export type Agent = {
  id?: string;
  name: string;
  agent_type?: string;
  status: string;
  last_heartbeat?: string;
  error_count?: number;
  last_error?: string | null;
};
