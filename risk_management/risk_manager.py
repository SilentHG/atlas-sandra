"""
ATLAS Risk Manager
==================
Validates proposed orders against account risk limits before
they reach the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from strategy_engine.base_strategy import TradeSignal, Signal


@dataclass
class RiskCheckResult:
    approved:       bool
    rejection_reason: str | None = None
    adjusted_qty:   float | None = None


class RiskManager:
    """
    Rule-based risk manager.

    Parameters
    ----------
    max_position_pct : float
        Max fraction of capital in a single position (default 5 %).
    max_drawdown_pct : float
        Circuit-breaker — halt trading if drawdown exceeds this (default 10 %).
    max_daily_loss   : float
        USD loss cap per day (default $500).
    risk_per_trade   : float
        Fraction of capital risked per trade for position sizing (default 1 %).
    """

    def __init__(
        self,
        capital:          float = 10_000.0,
        max_position_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        max_daily_loss:   float = 500.0,
        risk_per_trade:   float = 0.01,
    ) -> None:
        self.capital          = capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss   = max_daily_loss
        self.risk_per_trade   = risk_per_trade

        self._daily_loss: float = 0.0
        self._peak_capital: float = capital

    # ── Public API ────────────────────────────────────────────

    def check_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        open_positions_value: float = 0.0,
    ) -> RiskCheckResult:
        """Return approval decision for a given trade signal."""

        if signal.signal == Signal.HOLD:
            return RiskCheckResult(approved=False, rejection_reason="Signal is HOLD — no action.")

        # 1. Circuit-breaker: daily loss limit
        if self._daily_loss >= self.max_daily_loss:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Daily loss limit reached (${self._daily_loss:.2f}).",
            )

        # 2. Drawdown circuit-breaker
        drawdown = 1 - (self.capital / self._peak_capital)
        if drawdown >= self.max_drawdown_pct:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Max drawdown breached ({drawdown:.1%}).",
            )

        # 3. Position concentration
        if open_positions_value / self.capital > self.max_position_pct:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Position exceeds max concentration ({self.max_position_pct:.0%}).",
            )

        # 4. Position sizing via fixed fractional
        qty = self._size_position(signal, current_price)
        if qty <= 0:
            return RiskCheckResult(approved=False, rejection_reason="Calculated quantity ≤ 0.")

        return RiskCheckResult(approved=True, adjusted_qty=qty)

    def record_trade_pnl(self, pnl: float) -> None:
        """Update running P&L and capital tracking."""
        self.capital     += pnl
        self._daily_loss -= min(pnl, 0)    # only accumulate losses
        self._peak_capital = max(self._peak_capital, self.capital)

    def reset_daily_stats(self) -> None:
        """Call at the start of each trading session."""
        self._daily_loss = 0.0
        logger.info("[risk_manager] Daily stats reset. Capital: ${:.2f}", self.capital)

    # ── Internals ─────────────────────────────────────────────

    def _size_position(self, signal: TradeSignal, price: float) -> float:
        """Fixed-fractional position sizing using stop-loss distance."""
        risk_dollars = self.capital * self.risk_per_trade
        if signal.stop_loss and price != signal.stop_loss:
            distance = abs(price - signal.stop_loss)
            qty = risk_dollars / distance
        else:
            # Fallback: 1 % capital / price
            qty = risk_dollars / price
        return round(qty, 6)
