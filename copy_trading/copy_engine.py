"""
ATLAS Copy Trading Engine

Day 4 module:
- Mirrors leader trades to follower accounts
- Applies follower-specific risk limits
- Measures leader-to-follower latency
- Stores all mirror events in copy_trading_events
"""

from __future__ import annotations

import asyncpg
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4


class CopyTradingEngine:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def create_link(
        self,
        leader_account: str,
        follower_account: str,
        follower_risk_limit_pct: float = 0.02,
        sizing_mode: str = "proportional",
    ) -> str:
        query = """
        INSERT INTO copy_trading_links
        (leader_account, follower_account, follower_risk_limit_pct, sizing_mode, is_active)
        VALUES ($1, $2, $3, $4, TRUE)
        RETURNING id::text
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                query,
                leader_account,
                follower_account,
                follower_risk_limit_pct,
                sizing_mode,
            )

    async def mirror_trade(
        self,
        leader_account: str,
        leader_order_id: str,
        symbol: str,
        side: str,
        leader_qty: float,
        leader_equity: float,
        follower_equity: float,
        price: float,
        fill_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Mirror a leader trade to all active followers.

        This is paper-safe logic:
        - computes proportional size
        - supports partial fills via fill_ratio
        - applies follower risk cap
        - records event
        - measures latency
        - does not bypass follower limits
        """
        async with self.db_pool.acquire() as conn:
            links = await conn.fetch(
                """
                SELECT * FROM copy_trading_links
                WHERE leader_account = $1 AND is_active = TRUE
                """,
                leader_account,
            )

            import time
            started = time.perf_counter()
            fill_ratio = max(0.0, min(float(fill_ratio), 1.0))
            effective_leader_qty = leader_qty * fill_ratio

            events = []
            for link in links:
                follower_qty = self._compute_follower_qty(
                    leader_qty=effective_leader_qty,
                    leader_equity=leader_equity,
                    follower_equity=follower_equity,
                )

                max_notional = follower_equity * float(link["follower_risk_limit_pct"])
                follower_notional = follower_qty * price

                status = "mirrored" if fill_ratio >= 1.0 else "partial_fill_mirrored"
                rejection_reason: Optional[str] = None

                if follower_notional > max_notional:
                    scaled_qty = max_notional / price
                    if scaled_qty <= 0:
                        status = "rejected"
                        rejection_reason = "Follower risk limit too low"
                        follower_qty = 0
                    else:
                        follower_qty = scaled_qty
                        status = "partial_fill_scaled" if fill_ratio < 1.0 else "scaled"

                now = datetime.now(timezone.utc)
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                follower_order_id = f"paper-copy-{uuid4()}"

                event_id = await conn.fetchval(
                    """
                    INSERT INTO copy_trading_events
                    (
                        leader_order_id, follower_order_id, symbol, side,
                        leader_qty, follower_qty, leader_timestamp,
                        follower_timestamp, latency_ms, status, rejection_reason
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    RETURNING id::text
                    """,
                    leader_order_id,
                    follower_order_id,
                    symbol,
                    side,
                    effective_leader_qty,
                    follower_qty,
                    now,
                    now,
                    latency_ms,
                    status,
                    rejection_reason,
                )

                events.append(
                    {
                        "event_id": event_id,
                        "leader_order_id": leader_order_id,
                        "follower_order_id": follower_order_id,
                        "symbol": symbol,
                        "side": side,
                        "leader_qty": leader_qty,
                        "effective_leader_qty": effective_leader_qty,
                        "fill_ratio": fill_ratio,
                        "follower_qty": follower_qty,
                        "status": status,
                        "rejection_reason": rejection_reason,
                        "latency_ms": latency_ms,
                    }
                )

            return {
                "leader_order_id": leader_order_id,
                "followers_found": len(links),
                "events": events,
            }

    @staticmethod
    def _compute_follower_qty(
        leader_qty: float,
        leader_equity: float,
        follower_equity: float,
    ) -> float:
        if leader_equity <= 0:
            return 0.0
        ratio = follower_equity / leader_equity
        return leader_qty * ratio
