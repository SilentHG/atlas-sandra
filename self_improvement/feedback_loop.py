"""
ATLAS Self-Improvement Feedback Loop

Day 5 module:
- Reads strategy performance records
- Builds structured feedback summaries
- Marks feedback_sent_at for learning loop traceability
- Provides data for future strategy mutation/improvement
"""

from __future__ import annotations

import asyncpg
from datetime import datetime, timezone
from typing import Any, Dict, List


class FeedbackLoop:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def record_performance(
        self,
        strategy_id: str,
        sharpe_ratio: float,
        total_pnl: float,
        win_rate: float,
        max_drawdown: float,
        total_trades: int,
    ) -> str:
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO strategy_performance
                (
                    strategy_id, sharpe_ratio, total_pnl, win_rate,
                    max_drawdown, total_trades
                )
                VALUES ($1,$2,$3,$4,$5,$6)
                RETURNING id::text
                """,
                strategy_id,
                sharpe_ratio,
                total_pnl,
                win_rate,
                max_drawdown,
                total_trades,
            )

    async def generate_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM strategy_performance
                WHERE feedback_sent_at IS NULL
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

            feedback_items = []
            for row in rows:
                summary = self._build_summary(dict(row))

                await conn.execute(
                    """
                    UPDATE strategy_performance
                    SET feedback_sent_at = $1,
                        feedback_summary = $2
                    WHERE id = $3
                    """,
                    datetime.now(timezone.utc),
                    summary,
                    row["id"],
                )

                feedback_items.append(
                    {
                        "performance_id": str(row["id"]),
                        "strategy_id": str(row["strategy_id"]) if row["strategy_id"] else None,
                        "feedback_summary": summary,
                    }
                )

            return feedback_items

    def _build_summary(self, row: Dict[str, Any]) -> str:
        sharpe = row.get("sharpe_ratio") or 0
        pnl = row.get("total_pnl") or 0
        win_rate = row.get("win_rate") or 0
        drawdown = row.get("max_drawdown") or 0
        trades = row.get("total_trades") or 0

        if trades < 20:
            quality = "insufficient_sample"
        elif sharpe >= 1.5 and pnl > 0 and drawdown > -0.15:
            quality = "strong_candidate"
        elif pnl > 0 and sharpe > 0:
            quality = "needs_optimization"
        else:
            quality = "reject_or_mutate"

        return (
            f"quality={quality}; sharpe={sharpe:.3f}; pnl={pnl:.2f}; "
            f"win_rate={win_rate:.3f}; max_drawdown={drawdown:.3f}; "
            f"total_trades={trades}. Recommended action: "
            f"{self._recommend_action(quality)}"
        )

    @staticmethod
    def _recommend_action(quality: str) -> str:
        if quality == "strong_candidate":
            return "paper_trade_or_deploy_to_validation"
        if quality == "needs_optimization":
            return "mutate_parameters_and_retest"
        if quality == "insufficient_sample":
            return "collect_more_trades_before_decision"
        return "reject_or_generate_new_variant"
