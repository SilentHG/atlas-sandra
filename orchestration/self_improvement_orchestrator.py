"""
ATLAS Self-Improvement Orchestrator — Day 5

Uses strategy performance records to decide whether a strategy should be:
- kept
- mutated
- rejected
- sent to paper validation
"""

from __future__ import annotations

from typing import Any, Dict, List
import asyncpg

from self_improvement.feedback_loop import FeedbackLoop
from strategy_mutation.mutation_engine import StrategyMutationEngine


class SelfImprovementOrchestrator:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.feedback = FeedbackLoop(db_pool)
        self.mutator = StrategyMutationEngine(db_pool)

    async def run_cycle(self, limit: int = 10) -> Dict[str, Any]:
        feedback_items = await self.feedback.generate_feedback(limit=limit)

        actions: List[Dict[str, Any]] = []

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, strategy_id, sharpe_ratio, total_pnl, win_rate,
                       max_drawdown, total_trades, feedback_summary
                FROM strategy_performance
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        for row in rows:
            strategy_id = str(row["strategy_id"]) if row["strategy_id"] else None
            if not strategy_id:
                continue

            decision = self._decide(dict(row))
            action = {
                "strategy_id": strategy_id,
                "decision": decision,
                "mutated_strategy_ids": [],
            }

            if decision == "mutate":
                result = await self.mutator.mutate_strategy(
                    strategy_id=strategy_id,
                    mutation_type="feedback_driven_parameter_variation",
                    variation_pct=15.0,
                    variants=2,
                )
                action["mutated_strategy_ids"] = result.get("mutated_strategy_ids", [])

            actions.append(action)

        return {
            "status": "ok",
            "feedback_items": feedback_items,
            "actions": actions,
            "actions_count": len(actions),
        }

    def _decide(self, row: Dict[str, Any]) -> str:
        sharpe = float(row.get("sharpe_ratio") or 0)
        pnl = float(row.get("total_pnl") or 0)
        win_rate = float(row.get("win_rate") or 0)
        drawdown = float(row.get("max_drawdown") or 0)
        trades = int(row.get("total_trades") or 0)

        if trades < 20:
            return "collect_more_data"

        if sharpe >= 1.5 and pnl > 0 and drawdown > -0.15:
            return "promote_to_paper_validation"

        if pnl > 0 and sharpe > 0 and win_rate >= 0.4:
            return "optimize"

        return "mutate"
