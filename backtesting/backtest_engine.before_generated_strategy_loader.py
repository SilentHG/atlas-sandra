"""
ATLAS Backtest Engine — backtesting/backtest_engine.py
=======================================================
Reads historical OHLCV from TimescaleDB, applies a strategy,
enforces a 70/30 backtest/Monte Carlo split with zero data
leakage, models slippage + commission, creates daily equity curves,
and saves to backtests table.
"""
from __future__ import annotations

import json, uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from database import connection as db

# ── Config ─────────────────────────────────────────────────────────────────────
SLIPPAGE_BASE    = 0.001    # 0.1% per trade
COMMISSION_USD   = 1.0      # $1 per trade
MIN_HISTORY_DAYS = 724
BACKTEST_FRAC    = 0.70
MONTE_CARLO_FRAC = 0.30
MONTE_CARLO_RUNS = 500

@dataclass
class BacktestResult:
    strategy_id:        str
    symbol:             str
    split:              str   # "backtest" | "monte_carlo"
    start_date:         datetime
    end_date:           datetime
    total_trades:       int   = 0
    winning_trades:     int   = 0
    losing_trades:      int   = 0
    gross_pnl:          float = 0.0
    net_pnl:            float = 0.0
    sharpe_ratio:       float = 0.0
    sortino_ratio:      float = 0.0
    max_drawdown:       float = 0.0
    win_rate:           float = 0.0
    profit_factor:      float = 0.0
    avg_win:            float = 0.0
    avg_loss:           float = 0.0
    avg_trade_duration: float = 0.0   # minutes
    initial_capital:    float = 10_000.0
    final_capital:      float = 0.0
    equity_curve:       list  = field(default_factory=list)
    parameters:         dict  = field(default_factory=dict)
    run_id:             str   = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def annualized_return(self) -> float:
        days = max((self.end_date - self.start_date).days, 1)

        if self.initial_capital <= 0:
            return 0.0

        r = self.net_pnl / self.initial_capital

        # portfolio lost 100% or more
        if (1 + r) <= 0:
            return -1.0

        try:
            return float((1 + r) ** (365 / days) - 1)
        except Exception:
            return 0.0

    @property
    def total_return(self) -> float:
        return self.net_pnl / self.initial_capital

class BacktestEngine:
    """Walk-forward backtester with strict split discipline."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self._capital = initial_capital

    # ── Data loading ──────────────────────────────────────────────────────────

    async def _load_ohlcv(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = await db.fetch(
            """SELECT timestamp,open,high,low,close,volume
               FROM market_data
               WHERE symbol=$1 AND timestamp BETWEEN $2 AND $3
               ORDER BY timestamp ASC""",
            symbol, start, end,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        for c in ("open","high","low","close","volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    # ── Splits ────────────────────────────────────────────────────────────────

    @staticmethod
    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Buyer requirement:
        - First 70% of historical time range is used for the initial backtest.
        - Remaining 30% of historical time range is reserved for Monte Carlo validation.

        Important: this must split by timestamp, not row count, because historical
        minute data can have uneven density after backfill/API pagination.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        start_ts = df["timestamp"].min()
        end_ts = df["timestamp"].max()
        cutoff = start_ts + (end_ts - start_ts) * BACKTEST_FRAC

        backtest_df = df[df["timestamp"] <= cutoff]
        monte_carlo_df = df[df["timestamp"] > cutoff]

        return backtest_df, monte_carlo_df

    # ── Signal generation ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Simple EMA crossover as the default signal layer.
        Strategy-specific code can override this via execute()."""
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        df["signal"]   = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"]  =  1
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"]  = -1
        df["pos_change"] = df["signal"].diff().fillna(0)
        return df

    @staticmethod
    def _daily_equity_curve(equity: list[dict]) -> list[dict]:
        """
        Convert per-bar/minute equity curve into one end-of-day equity point.
        """
        if not equity:
            return []

        eq_df = pd.DataFrame(equity)
        eq_df["t"] = pd.to_datetime(eq_df["t"], utc=True, errors="coerce")
        eq_df["v"] = pd.to_numeric(eq_df["v"], errors="coerce")
        eq_df = eq_df.dropna(subset=["t", "v"])

        if eq_df.empty:
            return []

        daily = (
            eq_df.set_index("t")
            .resample("1D")
            .last()
            .dropna()
            .reset_index()
        )

        return [
            {"t": r["t"].date().isoformat(), "v": round(float(r["v"]), 2)}
            for _, r in daily.iterrows()
        ]

    @staticmethod
    def _monte_carlo_from_equity(
        equity_curve: list[dict],
        runs: int = MONTE_CARLO_RUNS,
    ) -> dict[str, Any]:
        """
        Monte Carlo validation using daily equity returns from the reserved 30%.
        Resamples daily returns to estimate drawdown and ending capital risk.
        """
        if len(equity_curve) < 30:
            return {
                "status": "partial",
                "reason": f"Insufficient daily equity points for Monte Carlo: {len(equity_curve)}",
                "runs": runs,
                "daily_points": len(equity_curve),
            }

        values = pd.Series([float(x["v"]) for x in equity_curve])
        returns = values.pct_change().dropna()

        if len(returns) < 20 or returns.std() == 0:
            return {
                "status": "partial",
                "reason": "Insufficient non-zero daily return distribution for Monte Carlo",
                "runs": runs,
                "daily_points": len(equity_curve),
            }

        start_capital = values.iloc[0]
        simulations = []
        max_drawdowns = []

        for _ in range(runs):
            sampled = returns.sample(n=len(returns), replace=True).reset_index(drop=True)
            path = start_capital * (1 + sampled).cumprod()
            peak = path.cummax()
            dd = ((path - peak) / peak).min()
            simulations.append(float(path.iloc[-1]))
            max_drawdowns.append(float(dd))

        return {
            "status": "pass",
            "runs": runs,
            "daily_points": len(equity_curve),
            "ending_capital_p05": round(float(np.percentile(simulations, 5)), 4),
            "ending_capital_p50": round(float(np.percentile(simulations, 50)), 4),
            "ending_capital_p95": round(float(np.percentile(simulations, 95)), 4),
            "max_drawdown_p05": round(float(np.percentile(max_drawdowns, 5)), 6),
            "max_drawdown_p50": round(float(np.percentile(max_drawdowns, 50)), 6),
            "max_drawdown_p95": round(float(np.percentile(max_drawdowns, 95)), 6),
        }

    # ── P&L simulation ────────────────────────────────────────────────────────

    def _simulate(self, df: pd.DataFrame, qty: float = 10.0) -> BacktestResult:
        capital    = self._capital
        peak       = capital
        equity     = []
        trades: list[dict] = []
        in_trade   = False
        entry_px   = 0.0
        entry_bar  = 0
        side       = 0

        for i, row in df.iterrows():
            equity.append({"t": str(row["timestamp"]), "v": round(capital, 2)})
            chg = row.get("pos_change", 0)

            if not in_trade and chg != 0:
                side     = int(np.sign(row["signal"]))
                slip     = row["close"] * SLIPPAGE_BASE * (1 + qty / 1000)
                entry_px = row["close"] + side * slip
                entry_bar = i
                in_trade  = True

            elif in_trade and (chg != 0 or i == df.index[-1]):
                slip     = row["close"] * SLIPPAGE_BASE * (1 + qty / 1000)
                exit_px  = row["close"] - side * slip
                gross_tr = side * (exit_px - entry_px) * qty
                net_tr   = gross_tr - COMMISSION_USD * 2
                capital += net_tr
                peak     = max(peak, capital)
                dur      = (i - entry_bar) if isinstance(i, int) else 1
                trades.append({
                    "gross": gross_tr, "net": net_tr, "bars": dur,
                    "win": net_tr > 0,
                })
                in_trade = False

                if chg != 0:
                    side     = int(np.sign(row["signal"]))
                    slip2    = row["close"] * SLIPPAGE_BASE * (1 + qty / 1000)
                    entry_px = row["close"] + side * slip2
                    entry_bar = i
                    in_trade  = True

        if not trades:
            return BacktestResult(
                strategy_id="", symbol="", split="",
                start_date=df["timestamp"].iloc[0],
                end_date=df["timestamp"].iloc[-1],
                initial_capital=self._capital,
                final_capital=capital,
                equity_curve=self._daily_equity_curve(equity),
            )

        wins   = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]
        gross  = sum(t["gross"] for t in trades)
        net    = sum(t["net"]   for t in trades)
        rets   = pd.Series([t["net"] for t in trades])

        sharpe  = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        neg     = rets[rets < 0]
        sortino = float(rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0.0

        eq_s = pd.Series([e["v"] for e in equity])
        roll_max = eq_s.cummax()
        dd = (eq_s - roll_max) / roll_max
        max_dd = float(dd.min())

        gross_wins  = sum(t["gross"] for t in wins)
        gross_loss  = abs(sum(t["gross"] for t in losses))
        pf = gross_wins / gross_loss if gross_loss > 0 else float("inf")

        r = BacktestResult(
            strategy_id="", symbol="", split="",
            start_date=df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1],
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            gross_pnl=round(gross, 4),
            net_pnl=round(net, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 6),
            win_rate=round(len(wins) / len(trades), 4),
            profit_factor=round(pf, 4),
            avg_win=round(np.mean([t["net"] for t in wins]), 4) if wins else 0.0,
            avg_loss=round(np.mean([t["net"] for t in losses]), 4) if losses else 0.0,
            avg_trade_duration=round(np.mean([t["bars"] for t in trades]), 2),
            initial_capital=self._capital,
            final_capital=round(capital, 4),
            equity_curve=self._daily_equity_curve(equity),
        )
        return r

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        symbol:      str,
        start:       datetime,
        end:         datetime,
        strategy_id: str = "",
        parameters:  dict | None = None,
        qty:         float = 10.0,
    ) -> dict[str, BacktestResult]:
        logger.info("[backtest] {} {} → {}", symbol, start.date(), end.date())
        # Buyer requirement: enforce at least 2 years of historical coverage.
        requested_days = (end - start).days
        if requested_days < MIN_HISTORY_DAYS:
            earliest = await db.fetchval(
                "SELECT MIN(timestamp) FROM market_data WHERE symbol=$1",
                symbol,
            )
            expanded_start = end - timedelta(days=MIN_HISTORY_DAYS)
            if earliest and earliest < expanded_start:
                start = earliest
            else:
                start = expanded_start

            logger.info(
                "[backtest] Expanded requested range to minimum history: {} → {}",
                start.date(), end.date(),
            )

        df = await self._load_ohlcv(symbol, start, end)
        if df.empty or len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")

        actual_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 0)
        if actual_days < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Minimum 2 years historical data required for {symbol}. "
                f"Found {actual_days} days from {df['timestamp'].min()} to {df['timestamp'].max()}."
            )

        df = self._compute_signals(df)
        backtest_df, monte_carlo_df = self._split(df)

        results: dict[str, BacktestResult] = {}
        for split_name, split_df in [
            ("backtest", backtest_df),
            ("monte_carlo", monte_carlo_df),
        ]:
            if len(split_df) < 30:
                continue

            r = self._simulate(split_df.reset_index(drop=True), qty=qty)
            r.strategy_id = strategy_id
            r.symbol      = symbol
            r.split       = split_name
            r.parameters  = parameters or {}

            if split_name == "monte_carlo":
                r.parameters = {
                    **(parameters or {}),
                    "monte_carlo": self._monte_carlo_from_equity(r.equity_curve),
                    "split_policy": "70% initial backtest / 30% Monte Carlo",
                }
            else:
                r.parameters = {
                    **(parameters or {}),
                    "split_policy": "70% initial backtest / 30% Monte Carlo",
                }

            results[split_name] = r
            logger.info(
                "[backtest] {} {} | trades={} net={:.2f} sharpe={:.3f} dd={:.2%}",
                split_name, symbol, r.total_trades, r.net_pnl, r.sharpe_ratio, r.max_drawdown,
            )
            await self._save(r)

        return results

    async def _save(self, r: BacktestResult) -> None:
        try:
            sid = uuid.UUID(r.strategy_id) if r.strategy_id else None
            await db.execute(
                """INSERT INTO backtests
                   (id,strategy_id,sharpe,sortino,max_drawdown,win_rate,
                    annualized_return,total_return,profit_factor,total_trades,
                    winning_trades,losing_trades,avg_win,avg_loss,
                    initial_capital,final_capital,start_date,end_date,
                    symbols,parameters,equity_curve,run_status,completed_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                           $15,$16,$17,$18,$19,$20::jsonb,$21::jsonb,'completed',NOW())""",
                uuid.UUID(r.run_id), sid,
                r.sharpe_ratio, r.sortino_ratio, r.max_drawdown, r.win_rate,
                r.annualized_return, r.total_return, r.profit_factor, r.total_trades,
                r.winning_trades, r.losing_trades, r.avg_win, r.avg_loss,
                r.initial_capital, r.final_capital, r.start_date, r.end_date,
                [r.symbol], json.dumps(r.parameters), json.dumps(r.equity_curve[-200:]),
            )
        except Exception as exc:
            logger.error("[backtest] Failed to save results: {}", exc)
