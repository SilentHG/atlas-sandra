"""
ATLAS Vectorized Backtester
============================
Runs a strategy against historical OHLCV data from market_data
and persists the results to the backtests table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from database import connection as db
from strategy_engine.base_strategy import BaseStrategy, Signal


@dataclass
class BacktestResult:
    strategy_name:       str
    start_date:          datetime
    end_date:            datetime
    symbols:             list[str]
    initial_capital:     float
    final_capital:       float       = 0.0
    total_return:        float       = 0.0
    annualized_return:   float       = 0.0
    sharpe_ratio:        float       = 0.0
    max_drawdown:        float       = 0.0
    win_rate:            float       = 0.0
    total_trades:        int         = 0
    winning_trades:      int         = 0
    losing_trades:       int         = 0
    avg_win:             float       = 0.0
    avg_loss:            float       = 0.0
    profit_factor:       float       = 0.0
    equity_curve:        list[dict]  = field(default_factory=list)


class Backtester:
    """Vectorized backtester for single-symbol strategies."""

    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10_000.0) -> None:
        self.strategy        = strategy
        self.initial_capital = initial_capital

    async def run(
        self,
        symbol:     str,
        start_date: datetime,
        end_date:   datetime,
    ) -> BacktestResult:
        """Fetch data, run strategy, compute metrics, and persist."""
        logger.info(
            "[backtester] Running '{}' on {} from {} to {}",
            self.strategy.name, symbol,
            start_date.date(), end_date.date(),
        )

        df = await self._load_data(symbol, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified range.")

        result = self._simulate(df, symbol, start_date, end_date)
        await self._persist(result)
        return result

    # ── Private ───────────────────────────────────────────────

    async def _load_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = await db.fetch(
            """
            SELECT time, open, high, low, close, volume
              FROM market_data
             WHERE symbol = $1
               AND time BETWEEN $2 AND $3
             ORDER BY time ASC
            """,
            symbol, start, end,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        return df

    def _simulate(
        self,
        df:         pd.DataFrame,
        symbol:     str,
        start_date: datetime,
        end_date:   datetime,
    ) -> BacktestResult:
        capital     = self.initial_capital
        position    = 0.0
        entry_price = 0.0
        trades: list[float] = []
        equity_curve: list[dict] = []

        for i in range(50, len(df)):
            window   = df.iloc[: i + 1].copy()
            sig      = self.strategy.generate_signal(symbol, window)
            price    = float(df.iloc[i]["close"])
            ts       = df.iloc[i]["time"]

            if sig.signal == Signal.BUY and position == 0.0:
                position    = (capital * 0.95) / price
                entry_price = price

            elif sig.signal in (Signal.SELL, Signal.CLOSE) and position > 0.0:
                pnl      = position * (price - entry_price)
                capital += pnl
                trades.append(pnl)
                position = 0.0

            portfolio_val = capital + position * price
            equity_curve.append({"time": str(ts), "equity": portfolio_val})

        # Close any open position at last price
        if position > 0.0:
            last_price = float(df.iloc[-1]["close"])
            pnl        = position * (last_price - entry_price)
            capital   += pnl
            trades.append(pnl)

        # ── Metrics ──────────────────────────────────────────
        returns         = pd.Series([e["equity"] for e in equity_curve]).pct_change().dropna()
        winning         = [t for t in trades if t > 0]
        losing          = [t for t in trades if t <= 0]
        equity_vals     = pd.Series([e["equity"] for e in equity_curve])
        roll_max        = equity_vals.cummax()
        drawdown_series = (equity_vals - roll_max) / roll_max
        max_dd          = float(drawdown_series.min())
        sharpe          = (
            float(returns.mean() / returns.std() * np.sqrt(252 * 390))
            if returns.std() > 0 else 0.0
        )

        return BacktestResult(
            strategy_name     = self.strategy.name,
            start_date        = start_date,
            end_date          = end_date,
            symbols           = [symbol],
            initial_capital   = self.initial_capital,
            final_capital     = capital,
            total_return      = (capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio      = sharpe,
            max_drawdown      = max_dd,
            win_rate          = len(winning) / len(trades) if trades else 0.0,
            total_trades      = len(trades),
            winning_trades    = len(winning),
            losing_trades     = len(losing),
            avg_win           = float(np.mean(winning)) if winning else 0.0,
            avg_loss          = float(np.mean(losing))  if losing  else 0.0,
            profit_factor     = (
                abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else 0.0
            ),
            equity_curve      = equity_curve,
        )

    async def _persist(self, result: BacktestResult) -> None:
        import json
        await db.execute(
            """
            INSERT INTO backtests
                (strategy_name, start_date, end_date, symbols,
                 initial_capital, final_capital, total_return,
                 sharpe_ratio, max_drawdown, win_rate,
                 total_trades, winning_trades, losing_trades,
                 avg_win, avg_loss, profit_factor,
                 equity_curve, status, completed_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                    $11,$12,$13,$14,$15,$16,$17::jsonb,'completed',NOW())
            """,
            result.strategy_name,
            result.start_date,
            result.end_date,
            result.symbols,
            result.initial_capital,
            result.final_capital,
            result.total_return,
            result.sharpe_ratio,
            result.max_drawdown,
            result.win_rate,
            result.total_trades,
            result.winning_trades,
            result.losing_trades,
            result.avg_win,
            result.avg_loss,
            result.profit_factor,
            json.dumps(result.equity_curve[-100:]),  # store last 100 points
        )
        logger.success(
            "[backtester] Saved results: return={:.1%}, sharpe={:.2f}, max_dd={:.1%}",
            result.total_return, result.sharpe_ratio, result.max_drawdown,
        )
