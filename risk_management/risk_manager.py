"""
ATLAS Risk Manager  ─ Day 2
============================
Validates proposed orders against hard risk limits before they reach the
execution layer.  Also tracks intra-day P&L and fires a **kill switch**
when the daily loss exceeds the configured cap.

Rules enforced
--------------
1. HOLD signal → always rejected (no-op gate).
2. Kill switch  → reject ALL new orders once daily loss ≥ 2 % of capital
                   (configurable via ``max_daily_loss_pct``).
3. Drawdown CB  → reject when peak-to-trough drawdown ≥ 10 % (configurable).
4. Position risk → reject when a single trade risks > 2 % of capital
                   (configurable via ``max_risk_per_trade``).
5. Concentration → reject when the open-position value in a single symbol
                   already exceeds ``max_position_pct`` of capital.
6. Sizing        → compute quantity via fixed-fractional (stop-distance method).

All state is in-process.  Snapshots are persisted to the ``risk_snapshots``
table (if it exists) so the dashboard can display live risk metrics.
If the table is absent the manager still works — the write just logs a warning.

Usage
-----
    from risk_management.risk_manager import RiskManager
    rm = RiskManager(capital=50_000.0)
    result = rm.check_signal(signal, current_price=193.40)

Credentials / env vars are NOT needed directly here; they come from
``config.settings`` which reads ``config/keys.env`` automatically.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from strategy_engine.base_strategy import Signal, TradeSignal


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class RiskCheckResult:
    """Result returned by :meth:`RiskManager.check_signal`."""
    approved:          bool
    rejection_reason:  str | None  = None
    adjusted_qty:      float | None = None
    risk_dollars:      float | None = None   # dollar risk of the proposed trade
    portfolio_risk_pct: float | None = None  # risk as % of current capital


@dataclass
class DailyStats:
    """Intra-day running totals, reset each session."""
    date:         date  = field(default_factory=lambda: date.today())
    realized_pnl: float = 0.0      # sum of closed trade P&L
    gross_loss:   float = 0.0      # cumulative loss side only (positive number)
    trade_count:  int   = 0
    kill_switch:  bool  = False    # latched True once daily loss cap is hit


# ── Risk Manager ──────────────────────────────────────────────────────────────


class RiskManager:
    """
    Rule-based, synchronous risk manager.

    Parameters
    ----------
    capital : float
        Current portfolio value in USD.
    max_risk_per_trade : float
        Maximum fraction of capital that can be risked on a single trade.
        Default = 0.02  (2 %).
    max_daily_loss_pct : float
        Kill-switch threshold: fraction of **opening** capital that can be
        lost in one session before all new orders are blocked.
        Default = 0.02  (2 %).
    max_drawdown_pct : float
        Peak-to-trough drawdown circuit-breaker (default 10 %).
    max_position_pct : float
        Max fraction of capital in a single symbol (concentration limit).
        Default = 0.05  (5 %).
    """

    def __init__(
        self,
        capital:            float = 10_000.0,
        max_risk_per_trade: float = 0.02,    # 2 % per-trade risk cap  ← new
        max_daily_loss_pct: float = 0.02,    # 2 % daily loss kill-switch ← new
        max_drawdown_pct:   float = 0.10,
        max_position_pct:   float = 0.05,
        # Legacy kwarg kept for backwards compat with existing tests
        risk_per_trade:     float | None = None,
        max_daily_loss:     float | None = None,   # absolute USD override
    ) -> None:
        self.capital            = capital
        self.max_risk_per_trade = risk_per_trade if risk_per_trade is not None else max_risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct   = max_drawdown_pct
        self.max_position_pct   = max_position_pct

        # Absolute daily loss cap (USD).  Can be overridden; else derived from pct.
        self._daily_loss_cap_usd: float = (
            max_daily_loss
            if max_daily_loss is not None
            else capital * max_daily_loss_pct
        )

        self._peak_capital: float = capital
        self._session_open: float = capital   # capital at start of today's session

        # Intra-day stats
        self._stats: DailyStats = DailyStats()

        # Backwards-compat alias used by original tests
        self._daily_loss: float = 0.0   # kept in sync with _stats.gross_loss

        logger.info(
            "[risk_mgr] Initialised | capital=${:,.0f} | risk/trade={:.0%} | "
            "daily-loss cap={:.0%} ({:.0f} USD) | drawdown CB={:.0%}",
            capital,
            self.max_risk_per_trade,
            max_daily_loss_pct,
            self._daily_loss_cap_usd,
            max_drawdown_pct,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        open_positions_value: float = 0.0,
        symbol_open_value: float = 0.0,
    ) -> RiskCheckResult:
        """
        Validate a trade signal against all risk rules.

        Parameters
        ----------
        signal : TradeSignal
        current_price : float
            Current mid-price of the instrument.
        open_positions_value : float
            Total USD value of all open positions (portfolio-wide).
        symbol_open_value : float
            USD value already deployed in *this specific* symbol.

        Returns
        -------
        RiskCheckResult
        """
        # ── 0. HOLD is a no-op ────────────────────────────────────────────────
        if signal.signal == Signal.HOLD:
            logger.info("[risk_mgr] REJECTED {}: Signal is HOLD", signal.symbol)
            return RiskCheckResult(
                approved=False,
                rejection_reason="Signal is HOLD — no action required.",
            )

        # ── 1. Kill switch — daily loss cap (2 % of session-open capital) ─────
        if self._stats.kill_switch or self._stats.gross_loss >= self._daily_loss_cap_usd:
            if not self._stats.kill_switch:
                self._engage_kill_switch()
            logger.critical(
                "[risk_mgr] REJECTED {}: Kill switch active — daily loss ${:.2f} >= cap ${:.2f}",
                signal.symbol, self._stats.gross_loss, self._daily_loss_cap_usd,
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"🔴 KILL SWITCH active — daily loss "
                    f"${self._stats.gross_loss:,.2f} ≥ cap "
                    f"${self._daily_loss_cap_usd:,.2f} "
                    f"({self.max_daily_loss_pct:.0%} of session capital)."
                ),
            )

        # ── 2. Drawdown circuit-breaker ───────────────────────────────────────
        drawdown = 1.0 - (self.capital / self._peak_capital) if self._peak_capital else 0.0
        if drawdown >= self.max_drawdown_pct:
            logger.warning(
                "[risk_mgr] REJECTED {}: Max drawdown {:.1%} >= {:.0%}",
                signal.symbol, drawdown, self.max_drawdown_pct,
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"Max drawdown breached: {drawdown:.1%} ≥ {self.max_drawdown_pct:.0%}."
                ),
            )

        # ── 3. Symbol concentration ───────────────────────────────────────────
        if self.capital > 0 and (symbol_open_value / self.capital) >= self.max_position_pct:
            logger.warning(
                "[risk_mgr] REJECTED {}: Concentration {:.1%} >= {:.0%} (max exposure)",
                signal.symbol, symbol_open_value / self.capital, self.max_position_pct,
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"Concentration limit hit for {signal.symbol}: "
                    f"{symbol_open_value / self.capital:.1%} ≥ {self.max_position_pct:.0%}. "
                    f"Position already at max exposure."
                ),
            )

        # ── 4. RISK-001: Max quantity hard limit (1000 shares) ─────────────────
        qty, risk_usd = self._size_position(signal, current_price)

        MAX_QTY = 1000.0
        if qty > MAX_QTY:
            logger.critical(
                "[risk_mgr] 🔴 RISK-001 REJECTED: {} qty={:.0f} exceeds max {} shares. "
                "Order blocked.",
                signal.symbol, qty, MAX_QTY,
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"RISK-001: Order quantity {qty:.0f} exceeds maximum "
                    f"allowed {MAX_QTY:.0f} shares per order."
                ),
            )

        # ── 5. Order value check for orders without explicit stop risk ────────
        order_value = qty * current_price
        max_order_value = self.capital * self.max_risk_per_trade
        if signal.stop_loss is None and order_value > max_order_value and self.capital > 0:
            logger.warning(
                "[risk_mgr] REJECTED {}: Order value ${:.2f} (qty={:.2f} × price={:.2f}) "
                "> 2%% of portfolio ${:.2f}",
                signal.symbol, order_value, qty, current_price, max_order_value,
            )
            # Try to cap quantity to stay within 2% limit
            capped_qty = max_order_value / current_price if current_price > 0 else 0
            if capped_qty <= 0:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=(
                        f"Order value ${order_value:,.2f} > {self.max_risk_per_trade:.0%} "
                        f"of portfolio (${max_order_value:,.2f}). Capped qty = 0."
                    ),
                )
            logger.info(
                "[risk_mgr] {} qty capped {:.4f} → {:.4f} to respect 2%% order value limit",
                signal.symbol, qty, capped_qty,
            )
            qty = capped_qty
            order_value = qty * current_price
            risk_usd = order_value

        # ── 6. Per-trade risk check (2 % cap) ─────────────────────────────────
        portfolio_risk_pct = risk_usd / self.capital if self.capital else 0.0
        if portfolio_risk_pct > self.max_risk_per_trade:
            # Try to reduce qty so risk equals exactly the cap
            capped_risk  = self.capital * self.max_risk_per_trade
            stop_dist    = abs(current_price - signal.stop_loss) if signal.stop_loss else current_price
            qty_capped   = capped_risk / stop_dist if stop_dist else 0.0
            if qty_capped <= 0:
                logger.warning(
                    "[risk_mgr] REJECTED {}: Per-trade risk {:.1%} > {:.0%}, capped qty=0",
                    signal.symbol, portfolio_risk_pct, self.max_risk_per_trade,
                )
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=(
                        f"Per-trade risk {portfolio_risk_pct:.1%} > limit "
                        f"{self.max_risk_per_trade:.0%}; qty capped to 0."
                    ),
                )
            logger.warning(
                "[risk_mgr] {} qty reduced {:.4f}→{:.4f} to respect {:.0%} risk cap",
                signal.symbol, qty, qty_capped, self.max_risk_per_trade,
            )
            qty      = qty_capped
            risk_usd = capped_risk
            portfolio_risk_pct = self.max_risk_per_trade

        if qty <= 0:
            logger.info("[risk_mgr] REJECTED {}: Calculated qty <= 0", signal.symbol)
            return RiskCheckResult(
                approved=False,
                rejection_reason="Calculated quantity ≤ 0 — trade skipped.",
            )

        logger.info(
            "[risk_mgr] ✅ APPROVED {} {} | qty={:.4f} | risk=${:.2f} ({:.2%} capital)",
            signal.signal.value, signal.symbol, qty, risk_usd, portfolio_risk_pct,
        )
        return RiskCheckResult(
            approved=True,
            adjusted_qty=round(qty, 6),
            risk_dollars=round(risk_usd, 2),
            portfolio_risk_pct=round(portfolio_risk_pct, 6),
        )

    def record_trade_pnl(self, pnl: float, symbol: str = "") -> None:
        """
        Call after each trade closes.  Updates capital, peak, and daily stats.
        Fires kill switch if the intra-day loss cap is breached.
        """
        self.capital              += pnl
        self._peak_capital         = max(self._peak_capital, self.capital)
        self._stats.realized_pnl  += pnl
        self._stats.trade_count   += 1

        if pnl < 0:
            loss = abs(pnl)
            self._stats.gross_loss += loss
            self._daily_loss        = self._stats.gross_loss   # backwards-compat alias

            logger.info(
                "[risk_mgr] Trade P&L: {}{:.2f} | session loss: ${:.2f} / cap ${:.2f} | capital: ${:,.2f}",
                "-" if pnl < 0 else "+", abs(pnl),
                self._stats.gross_loss, self._daily_loss_cap_usd,
                self.capital,
            )

            if self._stats.gross_loss >= self._daily_loss_cap_usd and not self._stats.kill_switch:
                self._engage_kill_switch()
        else:
            logger.info(
                "[risk_mgr] Trade P&L: +{:.2f} | capital: ${:,.2f}",
                pnl, self.capital,
            )

    def reset_daily_stats(self) -> None:
        """Call at the start of each trading session (market open)."""
        prev_loss    = self._stats.gross_loss
        prev_trades  = self._stats.trade_count
        self._stats  = DailyStats(date=date.today())
        self._daily_loss          = 0.0
        self._session_open        = self.capital
        self._daily_loss_cap_usd  = self.capital * self.max_daily_loss_pct

        logger.info(
            "[risk_mgr] Daily reset | prev session: loss=${:.2f} trades={} | "
            "new cap=${:.2f} ({:.0%} of ${:,.0f})",
            prev_loss, prev_trades,
            self._daily_loss_cap_usd,
            self.max_daily_loss_pct,
            self.capital,
        )

    # ── Status helpers ────────────────────────────────────────────────────────

    @property
    def kill_switch_active(self) -> bool:
        """True when trading is halted due to daily loss limit."""
        return self._stats.kill_switch

    @property
    def daily_pnl(self) -> float:
        """Net session P&L (positive = profit)."""
        return self._stats.realized_pnl

    @property
    def daily_loss(self) -> float:
        """Cumulative session loss (positive number, losses only)."""
        return self._stats.gross_loss

    def snapshot(self) -> dict[str, Any]:
        """Return a dict of current risk metrics suitable for logging / DB."""
        drawdown = 1.0 - (self.capital / self._peak_capital) if self._peak_capital else 0.0
        return {
            "timestamp":          datetime.now(tz=timezone.utc).isoformat(),
            "capital":            round(self.capital, 2),
            "peak_capital":       round(self._peak_capital, 2),
            "session_open":       round(self._session_open, 2),
            "drawdown_pct":       round(drawdown, 6),
            "daily_pnl":          round(self._stats.realized_pnl, 2),
            "daily_loss":         round(self._stats.gross_loss, 2),
            "daily_loss_cap_usd": round(self._daily_loss_cap_usd, 2),
            "daily_loss_cap_pct": self.max_daily_loss_pct,
            "trade_count":        self._stats.trade_count,
            "kill_switch":        self._stats.kill_switch,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _engage_kill_switch(self) -> None:
        self._stats.kill_switch = True
        logger.critical(
            "[risk_mgr] 🔴 KILL SWITCH ENGAGED — daily loss ${:.2f} hit cap "
            "${:.2f} ({:.0%}). ALL new orders blocked until session reset.",
            self._stats.gross_loss,
            self._daily_loss_cap_usd,
            self.max_daily_loss_pct,
        )

    def _size_position(self, signal: TradeSignal, price: float) -> tuple[float, float]:
        """
        Fixed-fractional sizing: risk ``risk_per_trade`` of capital.

        Returns (quantity, risk_in_usd).
        """
        risk_dollars = self.capital * self.max_risk_per_trade
        stop_distance = abs(price - signal.stop_loss) if signal.stop_loss and signal.stop_loss != price else None

        if stop_distance and stop_distance > 0:
            qty = risk_dollars / stop_distance
        else:
            # Fallback: use max_risk_per_trade % of capital as notional
            qty = risk_dollars / price if price else 0.0

        actual_risk = qty * (stop_distance if stop_distance else price)
        return qty, actual_risk

    # ── Async DB snapshot writer ───────────────────────────────────────────────

    async def persist_snapshot(self) -> None:
        """
        Attempt to write the current risk snapshot to the DB.
        Silently skips if the risk_snapshots table doesn't exist yet.
        """
        try:
            from database import connection as db
            snap = self.snapshot()
            await db.execute(
                """
                INSERT INTO risk_snapshots
                    (timestamp, capital, peak_capital, drawdown_pct,
                     daily_pnl, daily_loss, daily_loss_cap_usd,
                     trade_count, kill_switch)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT DO NOTHING
                """,
                snap["timestamp"],
                snap["capital"],
                snap["peak_capital"],
                snap["drawdown_pct"],
                snap["daily_pnl"],
                snap["daily_loss"],
                snap["daily_loss_cap_usd"],
                snap["trade_count"],
                snap["kill_switch"],
            )
        except Exception as exc:
            logger.debug("[risk_mgr] Snapshot persist skipped: {}", exc)

    def __repr__(self) -> str:
        return (
            f"<RiskManager capital=${self.capital:,.2f} "
            f"daily_loss=${self._stats.gross_loss:.2f} "
            f"kill_switch={self._stats.kill_switch}>"
        )

    # ── Async validate_order (pre-Alpaca gate) ────────────────────────────────

    async def validate_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: float,
        strategy_id: str | None = None,
    ) -> RiskCheckResult:
        """
        Async order validation that checks kill switch (DB) and risk limits
        BEFORE the order reaches Alpaca. This is the primary pre-trade gate.
        """
        try:
            from risk_management.kill_switch import get_kill_switch
            ks = get_kill_switch()

            # Check global kill switch from DB
            if await ks.is_armed():
                state = await ks.get_global_state()
                reason = f"Kill switch armed: {state.get('reason', 'unknown')}"
                logger.critical("[risk_mgr] REJECTED {} order: {}", symbol, reason)
                return RiskCheckResult(approved=False, rejection_reason=reason)

            # Check per-strategy kill switch
            if strategy_id and await ks.is_strategy_killed(strategy_id):
                reason = f"Strategy {strategy_id} is killed"
                logger.warning("[risk_mgr] REJECTED {} order: {}", symbol, reason)
                return RiskCheckResult(approved=False, rejection_reason=reason)

            # Check qty hard limit
            if qty > 1000:
                reason = f"RISK-001: qty {qty} > 1000 max"
                logger.critical("[risk_mgr] REJECTED {}: {}", symbol, reason)
                return RiskCheckResult(approved=False, rejection_reason=reason)

            # Check order value > 2% of portfolio
            order_value = qty * price
            max_value = self.capital * self.max_risk_per_trade
            if order_value > max_value and self.capital > 0:
                reason = f"Order value ${order_value:,.2f} > 2% of portfolio ${max_value:,.2f}"
                logger.warning("[risk_mgr] REJECTED {}: {}", symbol, reason)
                return RiskCheckResult(approved=False, rejection_reason=reason)

            # Check max exposure for this symbol
            from database import connection as db_conn
            existing = await db_conn.fetchval(
                "SELECT COALESCE(SUM(quantity * current_price), 0) FROM positions WHERE symbol=$1 AND status='open'",
                symbol,
            )
            if existing and self.capital > 0:
                total_exposure = float(existing) + order_value
                if total_exposure / self.capital >= self.max_position_pct:
                    reason = (
                        f"Max exposure for {symbol}: existing ${float(existing):,.2f} + "
                        f"new ${order_value:,.2f} = {total_exposure/self.capital:.1%} >= {self.max_position_pct:.0%}"
                    )
                    logger.warning("[risk_mgr] REJECTED: {}", reason)
                    return RiskCheckResult(approved=False, rejection_reason=reason)

            logger.info(
                "[risk_mgr] ✅ Pre-trade check PASSED: {} {} {} @ ${:.2f}",
                side, qty, symbol, price,
            )
            return RiskCheckResult(
                approved=True,
                adjusted_qty=qty,
                risk_dollars=order_value,
                portfolio_risk_pct=order_value / self.capital if self.capital else 0,
            )

        except Exception as exc:
            logger.error("[risk_mgr] validate_order error: {}", exc)
            return RiskCheckResult(approved=False, rejection_reason=f"Validation error: {exc}")
