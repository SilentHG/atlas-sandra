"""
ATLAS Kill Switch  — risk_management/kill_switch.py
=====================================================
Monitors portfolio P&L every second and enforces:
  KILL-001  daily loss  > 2 %   → halt ALL trading
  KILL-002  weekly loss > 4 %   → halt ALL trading
  KILL-003  manual halt via API → halt ALL trading
  KILL-004  state persists in DB (survives restart)
  KILL-005  per-strategy kill   → halt ONE strategy

Kill switch state is canonical in the `kill_switch_state` DB table.
No in-memory override is possible once armed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

from database import connection as db

# ── Constants ──────────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT  = 0.02   # KILL-001
WEEKLY_LOSS_LIMIT_PCT = 0.04   # KILL-002
POLL_INTERVAL_S       = 1      # check every second

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kill_switch_state (
    id              TEXT        PRIMARY KEY DEFAULT 'global',
    armed           BOOLEAN     NOT NULL DEFAULT FALSE,
    reason          TEXT,
    armed_at        TIMESTAMPTZ,
    armed_by        TEXT,
    daily_loss_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,
    weekly_loss_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    capital         DOUBLE PRECISION NOT NULL DEFAULT 0,
    scope           TEXT        NOT NULL DEFAULT 'portfolio',
    strategy_id     UUID,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_kill_state (
    strategy_id     TEXT        PRIMARY KEY,
    armed           BOOLEAN     NOT NULL DEFAULT FALSE,
    reason          TEXT,
    armed_at        TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class KillSwitch:
    """
    Persistent kill switch.  All state lives in the database so it
    survives process restarts (KILL-004).
    """

    def __init__(self, capital: float = 100_000.0) -> None:
        self._capital        = capital
        self._monitoring     = False
        self._cancel_cb: list = []   # callbacks called when armed

    # ── Public API ────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Ensure DB tables exist and load persisted state."""
        await db.execute(_ENSURE_TABLE_SQL)
        # Ensure global row exists
        await db.execute(
            """
            INSERT INTO kill_switch_state (id, capital, scope)
            VALUES ('global', $1, 'portfolio')
            ON CONFLICT (id) DO UPDATE SET capital = $1, updated_at = NOW()
            """,
            self._capital,
        )
        logger.info("[kill_switch] Initialised (capital=${:,.0f})", self._capital)
        # KILL-004: Restore state from DB on startup
        state = await self.get_global_state()
        if state["armed"]:
            logger.critical(
                "[kill_switch] ⚠ STATE RESTORED FROM DB — PREVIOUSLY ARMED\n"
                "  Reason: {}\n  Armed at: {}\n  Daily loss: ${:,.2f}\n  Weekly loss: ${:,.2f}",
                state.get("reason", "unknown"),
                state.get("armed_at", "unknown"),
                float(state.get("daily_loss_usd", 0)),
                float(state.get("weekly_loss_usd", 0)),
            )
        else:
            logger.info("[kill_switch] DB state: DISARMED — trading allowed")

    async def is_armed(self) -> bool:
        """True if the global kill switch is currently active."""
        row = await db.fetchrow("SELECT armed FROM kill_switch_state WHERE id='global'")
        return bool(row["armed"]) if row else False

    async def is_strategy_killed(self, strategy_id: str) -> bool:
        """True if this specific strategy has been halted (KILL-005)."""
        row = await db.fetchrow(
            "SELECT armed FROM strategy_kill_state WHERE strategy_id=$1", strategy_id
        )
        return bool(row["armed"]) if row else False

    async def get_global_state(self) -> dict[str, Any]:
        row = await db.fetchrow("SELECT * FROM kill_switch_state WHERE id='global'")
        if not row:
            return {"armed": False, "reason": None}
        return dict(row)

    async def arm(self, reason: str, armed_by: str = "system") -> None:
        """Arm the global kill switch and persist to DB."""
        await db.execute(
            """
            UPDATE kill_switch_state
               SET armed=TRUE, reason=$1, armed_at=NOW(), armed_by=$2, updated_at=NOW()
             WHERE id='global'
            """,
            reason,
            armed_by,
        )
        logger.critical("[kill_switch] 🔴 ARMED: {} (by={})", reason, armed_by)
        for cb in self._cancel_cb:
            try:
                await cb(reason)
            except Exception as exc:
                logger.warning("[kill_switch] Cancel callback error: {}", exc)

    async def disarm(self, disarmed_by: str = "manual") -> None:
        """Disarm the kill switch (manual override only)."""
        await db.execute(
            """
            UPDATE kill_switch_state
               SET armed=FALSE, reason=NULL, armed_at=NULL, updated_at=NOW()
             WHERE id='global'
            """,
        )
        logger.warning("[kill_switch] 🟢 DISARMED by {}", disarmed_by)

    async def kill_strategy(self, strategy_id: str, reason: str) -> None:
        """Halt one strategy only (KILL-005)."""
        await db.execute(
            """
            INSERT INTO strategy_kill_state (strategy_id, armed, reason, armed_at)
            VALUES ($1, TRUE, $2, NOW())
            ON CONFLICT (strategy_id) DO UPDATE
                SET armed=TRUE, reason=$2, armed_at=NOW(), updated_at=NOW()
            """,
            strategy_id,
            reason,
        )
        logger.warning("[kill_switch] Strategy {} killed: {}", strategy_id, reason)

    async def revive_strategy(self, strategy_id: str) -> None:
        """Re-enable a per-strategy kill."""
        await db.execute(
            "UPDATE strategy_kill_state SET armed=FALSE, updated_at=NOW() WHERE strategy_id=$1",
            strategy_id,
        )
        logger.info("[kill_switch] Strategy {} revived", strategy_id)

    def on_arm(self, callback) -> None:
        """Register an async callback to invoke when kill switch arms."""
        self._cancel_cb.append(callback)

    # ── P&L recording ────────────────────────────────────────────────────────

    async def record_pnl(self, daily_loss_usd: float, weekly_loss_usd: float) -> None:
        """Update loss figures and check thresholds."""
        await db.execute(
            """
            UPDATE kill_switch_state
               SET daily_loss_usd=$1, weekly_loss_usd=$2, updated_at=NOW()
             WHERE id='global'
            """,
            daily_loss_usd,
            weekly_loss_usd,
        )
        cap = self._capital or 1
        if daily_loss_usd / cap >= DAILY_LOSS_LIMIT_PCT:
            if not await self.is_armed():
                await self.arm(f"KILL-001: daily loss ${daily_loss_usd:,.2f} >= {DAILY_LOSS_LIMIT_PCT*100}% cap")
        elif weekly_loss_usd / cap >= WEEKLY_LOSS_LIMIT_PCT:
            if not await self.is_armed():
                await self.arm(f"KILL-002: weekly loss ${weekly_loss_usd:,.2f} >= {WEEKLY_LOSS_LIMIT_PCT*100}% cap")

    # ── Monitoring loop ───────────────────────────────────────────────────────

    async def start_monitor(self) -> None:
        """Continuously poll DB P&L and enforce limits."""
        self._monitoring = True
        logger.info("[kill_switch] Monitor started ({}s interval)", POLL_INTERVAL_S)
        while self._monitoring:
            try:
                await self._check_pnl()
            except Exception as exc:
                logger.error("[kill_switch] Monitor error: {}", exc)
            await asyncio.sleep(POLL_INTERVAL_S)

    async def stop_monitor(self) -> None:
        self._monitoring = False

    async def _check_pnl(self) -> None:
        row = await db.fetchrow(
            """
            SELECT
                COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) AS daily_loss
            FROM positions
            WHERE opened_at >= NOW() - INTERVAL '1 day'
              AND status IN ('open','closed')
            """
        )
        if not row:
            return
        daily_loss = float(row["daily_loss"] or 0)

        weekly_row = await db.fetchrow(
            """
            SELECT COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) AS weekly_loss
            FROM positions
            WHERE opened_at >= NOW() - INTERVAL '7 days'
            """
        )
        weekly_loss = float((weekly_row or {}).get("weekly_loss", 0))
        await self.record_pnl(daily_loss, weekly_loss)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: KillSwitch | None = None


def get_kill_switch(capital: float = 100_000.0) -> KillSwitch:
    global _instance
    if _instance is None:
        _instance = KillSwitch(capital=capital)
    return _instance
