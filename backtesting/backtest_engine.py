"""
ATLAS Backtest Engine — backtesting/backtest_engine.py
=======================================================
Reads historical OHLCV from TimescaleDB, applies a strategy,
enforces train/test/holdout splits (60/20/20) with zero data
leakage, models slippage + commission, and saves to backtests table.
"""
from __future__ import annotations

import json, uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from database import connection as db

# ── Config ─────────────────────────────────────────────────────────────────────
SLIPPAGE_BASE    = 0.001    # 0.1% per trade
COMMISSION_USD   = 1.0      # $1 per trade
TRAIN_FRAC       = 0.60
TEST_FRAC        = 0.20
# holdout = remaining 20%

@dataclass
class BacktestResult:
    strategy_id:        str
    symbol:             str
    split:              str   # "train" | "test" | "holdout"
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
        r    = self.net_pnl / self.initial_capital
        return (1 + r) ** (365 / days) - 1

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
    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n     = len(df)
        i_tr  = int(n * TRAIN_FRAC)
        i_te  = int(n * (TRAIN_FRAC + TEST_FRAC))
        return df.iloc[:i_tr], df.iloc[i_tr:i_te], df.iloc[i_te:]

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
                equity_curve=equity[-200:],
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
            equity_curve=equity[-200:],
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
        df = await self._load_ohlcv(symbol, start, end)
        if df.empty or len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")

        df = self._compute_signals(df)
        train_df, test_df, holdout_df = self._split(df)

        results: dict[str, BacktestResult] = {}
        for split_name, split_df in [
            ("train", train_df), ("test", test_df), ("holdout", holdout_df)
        ]:
            if len(split_df) < 5:
                continue
            r = self._simulate(split_df.reset_index(drop=True), qty=qty)
            r.strategy_id = strategy_id
            r.symbol      = symbol
            r.split       = split_name
            r.parameters  = parameters or {}
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
