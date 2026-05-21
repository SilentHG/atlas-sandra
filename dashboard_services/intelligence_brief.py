"""
ATLAS Daily Intelligence Brief

Day 4 module:
- Produces a non-technical market/system summary
- Pulls risk state, feature freshness, strategy performance, and anomalies
- Stores generated brief in intelligence_briefs table
"""

from __future__ import annotations

import asyncpg
import json
from datetime import datetime, timezone
from typing import Any, Dict, List


class IntelligenceBriefService:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def generate_brief(self) -> Dict[str, Any]:
        async with self.db_pool.acquire() as conn:
            market_regime = await self._get_market_regime(conn)
            risk_summary = await self._get_risk_summary(conn)
            top_performers = await self._get_top_strategy_performers(conn)
            anomalies = await self._get_anomalies(conn)
            active_signals = await self._get_active_signals(conn)

            brief_text = self._compose_text(
                market_regime=market_regime,
                risk_summary=risk_summary,
                top_performers=top_performers,
                anomalies=anomalies,
                active_signals=active_signals,
            )

            brief_id = await conn.fetchval(
                """
                INSERT INTO intelligence_briefs
                (
                    market_regime, active_signals, anomalies,
                    risk_summary, top_strategy_performers, brief_text
                )
                VALUES ($1,$2::jsonb,$3::jsonb,$4::jsonb,$5::jsonb,$6)
                RETURNING id::text
                """,
                market_regime,
                json.dumps(active_signals),
                json.dumps(anomalies),
                json.dumps(risk_summary),
                json.dumps(top_performers),
                brief_text,
            )

            return {
                "brief_id": brief_id,
                "market_regime": market_regime,
                "active_signals": active_signals,
                "anomalies": anomalies,
                "risk_summary": risk_summary,
                "top_strategy_performers": top_performers,
                "brief_text": brief_text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    async def _get_market_regime(self, conn: asyncpg.Connection) -> str:
        row = await conn.fetchrow(
            """
            SELECT value
            FROM feature_store
            WHERE feature_name ILIKE '%regime%'
            ORDER BY computed_at DESC
            LIMIT 1
            """
        )
        if not row:
            return "unknown"
        value = row["value"]
        if value is None:
            return "unknown"
        if value >= 1:
            return "trending_or_risk_on"
        if value <= -1:
            return "risk_off_or_high_volatility"
        return "neutral"

    async def _get_risk_summary(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM kill_switch_state
            ORDER BY updated_at DESC NULLS LAST
            LIMIT 1
            """
        )
        if not row:
            return {"kill_switch_armed": False, "note": "No kill-switch state found"}

        return {
            "kill_switch_armed": bool(row.get("is_armed", False)) if hasattr(row, "get") else False,
            "reason": row.get("reason") if hasattr(row, "get") else None,
        }

    async def _get_top_strategy_performers(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        rows = await conn.fetch(
            """
            SELECT strategy_id::text, sharpe_ratio, total_pnl, win_rate, max_drawdown
            FROM strategy_performance
            ORDER BY total_pnl DESC NULLS LAST
            LIMIT 5
            """
        )
        return [dict(row) for row in rows]

    async def _get_anomalies(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        rows = await conn.fetch(
            """
            SELECT symbol, feature_name, value, computed_at
            FROM feature_store
            WHERE feature_name ILIKE '%anomaly%'
            ORDER BY computed_at DESC
            LIMIT 10
            """
        )
        return [dict(row) for row in rows]

    async def _get_active_signals(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        rows = await conn.fetch(
            """
            SELECT symbol, feature_name, value, computed_at
            FROM feature_store
            WHERE computed_at > now() - interval '10 minutes'
            ORDER BY computed_at DESC
            LIMIT 10
            """
        )
        return [dict(row) for row in rows]

    def _compose_text(
        self,
        market_regime: str,
        risk_summary: Dict[str, Any],
        top_performers: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]],
        active_signals: List[Dict[str, Any]],
    ) -> str:
        risk_state = "armed" if risk_summary.get("kill_switch_armed") else "disarmed"
        return (
            f"ATLAS Daily Brief: Current market regime is {market_regime}. "
            f"The portfolio kill switch is {risk_state}. "
            f"There are {len(active_signals)} recent feature/signal updates, "
            f"{len(anomalies)} anomaly readings, and {len(top_performers)} tracked strategy performers. "
            "Review validation reports before deploying any new strategy to paper trading."
        )
