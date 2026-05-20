"""
ATLAS Walk-Forward Validator — backtesting/walk_forward.py
==========================================================
Runs rolling train/test windows across all available data.
Proves strategy is not curve-fitted to a single period.

Window logic:
  - train_days:  how many days to train on (default 30)
  - test_days:   how many days to test on  (default 7)
  - step_days:   how many days to slide forward each iteration (default 7)

Stability score = % of windows with positive net_pnl on test set.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from database import connection as db
from backtesting.backtest_engine import BacktestEngine, BacktestResult


@dataclass
class WalkForwardWindow:
    window_id:    int
    train_start:  datetime
    train_end:    datetime
    test_start:   datetime
    test_end:     datetime
    train_result: BacktestResult | None = None
    test_result:  BacktestResult | None = None

    @property
    def profitable(self) -> bool:
        return self.test_result is not None and self.test_result.net_pnl > 0

    @property
    def test_sharpe(self) -> float:
        return self.test_result.sharpe_ratio if self.test_result else 0.0

    @property
    def test_pnl(self) -> float:
        return self.test_result.net_pnl if self.test_result else 0.0


@dataclass
class WalkForwardReport:
    strategy_id:      str
    symbol:           str
    run_id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    train_days:       int = 30
    test_days:        int = 7
    step_days:        int = 7
    windows:          list[WalkForwardWindow] = field(default_factory=list)

    # Aggregate stats
    total_windows:    int   = 0
    profitable_windows: int = 0
    stability_score:  float = 0.0   # % windows profitable
    avg_test_sharpe:  float = 0.0
    avg_test_pnl:     float = 0.0
    total_test_pnl:   float = 0.0
    best_window_pnl:  float = 0.0
    worst_window_pnl: float = 0.0
    pnl_series:       list  = field(default_factory=list)  # per-window test pnl
    recommendation:   str   = ""

    def compute_aggregates(self) -> None:
        if not self.windows:
            return
        self.total_windows      = len(self.windows)
        self.profitable_windows = sum(1 for w in self.windows if w.profitable)
        self.stability_score    = round(self.profitable_windows / self.total_windows, 4)
        sharpes  = [w.test_sharpe for w in self.windows if w.test_result]
        pnls     = [w.test_pnl    for w in self.windows if w.test_result]
        self.avg_test_sharpe  = round(float(np.mean(sharpes)), 4) if sharpes else 0.0
        self.avg_test_pnl     = round(float(np.mean(pnls)),    4) if pnls    else 0.0
        self.total_test_pnl   = round(float(np.sum(pnls)),     4) if pnls    else 0.0
        self.best_window_pnl  = round(float(np.max(pnls)),     4) if pnls    else 0.0
        self.worst_window_pnl = round(float(np.min(pnls)),     4) if pnls    else 0.0
        self.pnl_series       = [round(p, 4) for p in pnls]

        # Recommendation
        if self.stability_score >= 0.65 and self.avg_test_sharpe >= 0.5:
            self.recommendation = "DEPLOY"
        elif self.stability_score >= 0.50:
            self.recommendation = "PAPER_TRADE"
        elif self.stability_score >= 0.35:
            self.recommendation = "OPTIMIZE"
        else:
            self.recommendation = "REJECT"


class WalkForwardValidator:
    """
    Runs walk-forward analysis on a strategy across a symbol.
    Uses existing BacktestEngine internals — no duplication.
    """

    def __init__(
        self,
        train_days: int = 30,
        test_days:  int = 7,
        step_days:  int = 7,
        qty:        float = 10.0,
    ) -> None:
        self._train_days = train_days
        self._test_days  = test_days
        self._step_days  = step_days
        self._qty        = qty
        self._engine     = BacktestEngine()

    async def run(
        self,
        strategy_id: str,
        symbol:      str,
        start:       datetime,
        end:         datetime,
        parameters:  dict | None = None,
    ) -> WalkForwardReport:

        report = WalkForwardReport(
            strategy_id=strategy_id,
            symbol=symbol,
            train_days=self._train_days,
            test_days=self._test_days,
            step_days=self._step_days,
        )

        logger.info(
            "[wf] Starting walk-forward | {} {} | train={}d test={}d step={}d",
            symbol, strategy_id[:8], self._train_days, self._test_days, self._step_days,
        )

        # Load all data once
        df = await self._engine._load_ohlcv(symbol, start, end)
        if df.empty or len(df) < 50:
            raise ValueError(f"[wf] Insufficient data for {symbol}: {len(df)} bars")

        df = self._engine._compute_signals(df)
        df = df.reset_index(drop=True)

        # Build windows
        window_id  = 0
        cursor     = start

        while True:
            train_start = cursor
            train_end   = cursor + timedelta(days=self._train_days)
            test_start  = train_end
            test_end    = train_end + timedelta(days=self._test_days)

            if test_end > end:
                break

            # Slice dataframes
            train_df = df[
                (df["timestamp"] >= train_start) &
                (df["timestamp"] <  train_end)
            ].reset_index(drop=True)

            test_df = df[
                (df["timestamp"] >= test_start) &
                (df["timestamp"] <  test_end)
            ].reset_index(drop=True)

            if len(train_df) < 10 or len(test_df) < 5:
                cursor += timedelta(days=self._step_days)
                continue

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # Run simulation on each split
            train_r = self._engine._simulate(train_df, qty=self._qty)
            train_r.strategy_id = strategy_id
            train_r.symbol      = symbol
            train_r.split       = f"wf_train_{window_id}"
            train_r.parameters  = parameters or {}

            test_r = self._engine._simulate(test_df, qty=self._qty)
            test_r.strategy_id = strategy_id
            test_r.symbol      = symbol
            test_r.split       = f"wf_test_{window_id}"
            test_r.parameters  = parameters or {}

            window.train_result = train_r
            window.test_result  = test_r
            report.windows.append(window)

            logger.info(
                "[wf] Window {:02d} | train_pnl={:.2f} | test_pnl={:.2f} | test_sharpe={:.3f} | {}",
                window_id,
                train_r.net_pnl,
                test_r.net_pnl,
                test_r.sharpe_ratio,
                "✅" if test_r.net_pnl > 0 else "❌",
            )

            window_id += 1
            cursor += timedelta(days=self._step_days)

        # Compute summary stats
        report.compute_aggregates()

        logger.info(
            "[wf] Done | windows={} | profitable={} | stability={:.0%} | "
            "avg_sharpe={:.3f} | total_pnl={:.2f} | recommendation={}",
            report.total_windows,
            report.profitable_windows,
            report.stability_score,
            report.avg_test_sharpe,
            report.total_test_pnl,
            report.recommendation,
        )

        # Save to DB
        await self._save(report)
        return report

    async def _save(self, report: WalkForwardReport) -> None:
        """Save walk-forward summary to backtests table."""
        try:
            sid = uuid.UUID(report.strategy_id) if report.strategy_id else None
            await db.execute(
                """INSERT INTO backtests
                   (id, strategy_id, sharpe, sortino, max_drawdown, win_rate,
                    annualized_return, total_return, profit_factor, total_trades,
                    winning_trades, losing_trades, avg_win, avg_loss,
                    initial_capital, final_capital, start_date, end_date,
                    symbols, parameters, equity_curve, run_status, completed_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                           $15,$16,$17,$18,$19,$20::jsonb,$21::jsonb,'completed',NOW())
                   ON CONFLICT DO NOTHING""",
                uuid.UUID(report.run_id),
                sid,
                report.avg_test_sharpe,
                0.0,
                0.0,
                report.stability_score,
                0.0,
                report.stability_score,
                0.0,
                report.total_windows,
                report.profitable_windows,
                report.total_windows - report.profitable_windows,
                report.best_window_pnl,
                report.worst_window_pnl,
                10_000.0,
                10_000.0 + report.total_test_pnl,
                report.windows[0].train_start if report.windows else datetime.now(timezone.utc),
                report.windows[-1].test_end   if report.windows else datetime.now(timezone.utc),
                [report.symbol],
                json.dumps({
                    "type":           "walk_forward",
                    "train_days":     report.train_days,
                    "test_days":      report.test_days,
                    "step_days":      report.step_days,
                    "recommendation": report.recommendation,
                    "pnl_series":     report.pnl_series,
                }),
                json.dumps([]),
            )
            logger.info("[wf] Report saved → run_id={}", report.run_id)
        except Exception as exc:
            logger.error("[wf] Failed to save report: {}", exc)
