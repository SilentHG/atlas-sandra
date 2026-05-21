"""
ATLAS Regime Testing

Day 4 module:
- Breaks strategy performance down by market regime
- Supports VAL-006 buyer test
- A robust strategy should survive bull, bear, sideways, high-vol, and low-vol conditions
"""

from __future__ import annotations

from typing import Any, Dict, List


class RegimeTester:
    REQUIRED_REGIMES = ["bull", "bear", "sideways", "high_vol", "low_vol"]

    def evaluate(self, regime_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        regimes = {}
        positive_or_neutral = 0
        missing = []

        for regime in self.REQUIRED_REGIMES:
            metrics = regime_metrics.get(regime)

            if metrics is None:
                missing.append(regime)
                regimes[regime] = {
                    "status": "missing",
                    "net_pnl": None,
                    "sharpe_ratio": None,
                }
                continue

            net_pnl = float(metrics.get("net_pnl") or 0)
            sharpe = float(metrics.get("sharpe_ratio") or 0)

            status = "positive_or_neutral" if net_pnl >= 0 or sharpe >= 0 else "negative"
            if status == "positive_or_neutral":
                positive_or_neutral += 1

            regimes[regime] = {
                "status": status,
                "net_pnl": net_pnl,
                "sharpe_ratio": sharpe,
                "total_trades": metrics.get("total_trades"),
                "max_drawdown": metrics.get("max_drawdown"),
            }

        if missing:
            overall_status = "partial"
            reason = f"Missing regime results: {missing}"
        elif positive_or_neutral >= 3:
            overall_status = "pass"
            reason = f"{positive_or_neutral}/5 regimes positive or neutral"
        else:
            overall_status = "fail"
            reason = f"Only {positive_or_neutral}/5 regimes positive or neutral"

        return {
            "status": overall_status,
            "positive_or_neutral_regimes": positive_or_neutral,
            "required_regimes": self.REQUIRED_REGIMES,
            "missing_regimes": missing,
            "reason": reason,
            "regimes": regimes,
        }
