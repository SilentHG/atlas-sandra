"""
ATLAS Strategy Mutation Engine — Day 5

Creates parameter variants from an existing strategy for retesting.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Dict, List

import asyncpg


class StrategyMutationEngine:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def mutate_strategy(
        self,
        strategy_id: str,
        mutation_type: str = "parameter_variation",
        variation_pct: float = 20.0,
        variants: int = 3,
    ) -> Dict[str, Any]:
        async with self.db_pool.acquire() as conn:
            strategy = await conn.fetchrow(
                """
                SELECT *
                FROM strategies
                WHERE id = $1::uuid
                """,
                strategy_id,
            )

            if not strategy:
                return {
                    "status": "fail",
                    "reason": "Strategy not found",
                    "strategy_id": strategy_id,
                    "mutated_strategy_ids": [],
                }

            base_params = strategy.get("parameters") if hasattr(strategy, "get") else strategy["parameters"]
            if isinstance(base_params, str):
                try:
                    base_params = json.loads(base_params)
                except Exception:
                    base_params = {}
            base_params = base_params or {}

            mutated_ids: List[str] = []

            for i in range(variants):
                mutated_params = self._mutate_params(
                    base_params=base_params,
                    variation_pct=variation_pct,
                    variant_index=i,
                )

                new_id = str(uuid.uuid4())
                new_name = f"{strategy['name']}_mutant_{i+1}_{new_id[:8]}"

                await conn.execute(
                    """
                    INSERT INTO strategies
                    (
                        id, name, description, strategy_type, symbols, timeframe,
                        parameters, risk_per_trade, max_position_size, status, code
                    )
                    VALUES ($1::uuid,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11)
                    """,
                    new_id,
                    new_name,
                    f"Mutated variant of {strategy['name']} using {mutation_type} at {variation_pct}% variation.",
                    strategy["strategy_type"],
                    strategy["symbols"],
                    strategy["timeframe"],
                    json.dumps(mutated_params),
                    strategy["risk_per_trade"],
                    strategy["max_position_size"],
                    "draft",
                    strategy["code"],
                )

                mutated_ids.append(new_id)

            return {
                "status": "ok",
                "source_strategy_id": strategy_id,
                "mutation_type": mutation_type,
                "variation_pct": variation_pct,
                "mutated_strategy_ids": mutated_ids,
            }

    def _mutate_params(
        self,
        base_params: Dict[str, Any],
        variation_pct: float,
        variant_index: int,
    ) -> Dict[str, Any]:
        params = deepcopy(base_params)
        factor = 1 + ((variation_pct / 100.0) * (variant_index + 1))

        if not params:
            params = {
                "fast_ema": 9,
                "slow_ema": 21,
                "rsi_period": 14,
                "atr_stop_multiplier": 2.0,
                "take_profit_r": 2.0,
            }

        for key, value in list(params.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if variant_index % 2 == 0:
                    params[key] = round(value * factor, 4)
                else:
                    params[key] = round(max(1, value / factor), 4)

        params["mutation_meta"] = {
            "variation_pct": variation_pct,
            "variant_index": variant_index,
        }
        return params
