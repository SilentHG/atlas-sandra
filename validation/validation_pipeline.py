"""
ATLAS Validation Pipeline

Day 4 module:
- Centralizes validation decisions after strategy generation/backtesting
- Combines train/test/holdout, walk-forward, sensitivity, regime, and risk checks
- Produces a deployment recommendation
- Stores validation reports in validation_reports table
"""

from __future__ import annotations

import asyncpg
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ValidationPipeline:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def validate_strategy(
        self,
        strategy_id: str,
        backtest_metrics: Dict[str, Any],
        walk_forward: Optional[Dict[str, Any]] = None,
        sensitivity: Optional[Dict[str, Any]] = None,
        regime_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        checks = {
            "backtest": self._check_backtest(backtest_metrics),
            "walk_forward": self._check_walk_forward(walk_forward or {}),
            "sensitivity": self._check_sensitivity(sensitivity or {}),
            "regime": self._check_regime(regime_results or {}),
        }

        passed = sum(1 for result in checks.values() if result["status"] == "pass")
        partial = sum(1 for result in checks.values() if result["status"] == "partial")
        failed = sum(1 for result in checks.values() if result["status"] == "fail")

        if failed == 0 and passed >= 3:
            recommendation = "DEPLOY_TO_PAPER"
            status = "pass"
        elif failed <= 1 and passed + partial >= 3:
            recommendation = "OPTIMIZE_AND_RETEST"
            status = "partial"
        else:
            recommendation = "REJECT"
            status = "fail"

        report = {
            "strategy_id": strategy_id,
            "status": status,
            "recommendation": recommendation,
            "checks": checks,
            "summary": {
                "passed": passed,
                "partial": partial,
                "failed": failed,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._save_report(strategy_id, "full_validation", status, report, recommendation)
        return report

    async def _save_report(
        self,
        strategy_id: str,
        validation_type: str,
        status: str,
        metrics: Dict[str, Any],
        recommendation: str,
    ) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO validation_reports
                (strategy_id, validation_type, status, metrics, recommendation)
                VALUES ($1,$2,$3,$4::jsonb,$5)
                """,
                strategy_id,
                validation_type,
                status,
                self._json(metrics),
                recommendation,
            )

    def _check_backtest(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        required = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
            "avg_trade_duration",
            "gross_pnl",
            "net_pnl",
            "slippage_per_trade",
            "commission_per_trade",
        ]

        missing = [key for key in required if metrics.get(key) is None]
        if missing:
            return {
                "status": "fail",
                "reason": f"Missing metrics: {missing}",
            }

        if metrics["total_trades"] < 20:
            return {
                "status": "partial",
                "reason": "Trade count below statistical confidence threshold",
            }

        if metrics["net_pnl"] <= 0 or metrics["profit_factor"] < 1:
            return {
                "status": "fail",
                "reason": "Strategy not profitable after costs",
            }

        if metrics["slippage_per_trade"] <= 0 or metrics["commission_per_trade"] <= 0:
            return {
                "status": "fail",
                "reason": "Trading costs are not modeled correctly",
            }

        return {
            "status": "pass",
            "reason": "Backtest metrics acceptable",
        }

    def _check_walk_forward(self, result: Dict[str, Any]) -> Dict[str, Any]:
        score = result.get("stability_score")
        if score is None:
            return {"status": "partial", "reason": "Walk-forward result unavailable"}

        if score >= 0.6:
            return {"status": "pass", "reason": f"Stability score {score:.2f} >= 0.60"}

        if score >= 0.4:
            return {"status": "partial", "reason": f"Stability score {score:.2f} requires optimization"}

        return {"status": "fail", "reason": f"Stability score {score:.2f} too low"}

    def _check_sensitivity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        drop = result.get("max_sharpe_drop_pct")
        if drop is None:
            return {"status": "partial", "reason": "Sensitivity result unavailable"}

        if drop <= 50:
            return {"status": "pass", "reason": f"Sharpe drop {drop:.1f}% within limit"}

        return {"status": "fail", "reason": f"Sharpe drop {drop:.1f}% indicates overfitting"}

    def _check_regime(self, result: Dict[str, Any]) -> Dict[str, Any]:
        regimes = result.get("regimes", {})
        if not regimes:
            return {"status": "partial", "reason": "Regime result unavailable"}

        positive_or_neutral = 0
        for metrics in regimes.values():
            pnl = metrics.get("net_pnl", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            if pnl >= 0 or sharpe >= 0:
                positive_or_neutral += 1

        if positive_or_neutral >= 3:
            return {
                "status": "pass",
                "reason": f"{positive_or_neutral}/5 regimes positive or neutral",
            }

        return {
            "status": "fail",
            "reason": f"Only {positive_or_neutral}/5 regimes positive or neutral",
        }

    @staticmethod
    def _json(payload: Dict[str, Any]) -> str:
        import json
        return json.dumps(payload, default=str)
