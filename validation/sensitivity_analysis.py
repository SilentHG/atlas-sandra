"""
ATLAS Sensitivity / Overfitting Analysis

Day 4 module:
- Tests whether a strategy is fragile to parameter changes
- Flags overfitting if Sharpe drops too much under +/- parameter variation
- Supports VAL-005 buyer test
"""

from __future__ import annotations

from typing import Any, Dict, List


class SensitivityAnalyzer:
    def __init__(self, variation_pct: float = 20.0):
        self.variation_pct = variation_pct

    def analyze(
        self,
        baseline_metrics: Dict[str, Any],
        variant_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        baseline_sharpe = float(baseline_metrics.get("sharpe_ratio") or 0)

        if baseline_sharpe == 0:
            return {
                "status": "partial",
                "variation_pct": self.variation_pct,
                "baseline_sharpe": baseline_sharpe,
                "max_sharpe_drop_pct": None,
                "is_overfit": None,
                "reason": "Baseline Sharpe is zero; sensitivity cannot be measured reliably",
                "variants": variant_metrics,
            }

        drops = []
        for variant in variant_metrics:
            variant_sharpe = float(variant.get("sharpe_ratio") or 0)
            drop_pct = ((baseline_sharpe - variant_sharpe) / abs(baseline_sharpe)) * 100
            drops.append(drop_pct)

        max_drop = max(drops) if drops else 0
        is_overfit = max_drop > 50

        return {
            "status": "fail" if is_overfit else "pass",
            "variation_pct": self.variation_pct,
            "baseline_sharpe": baseline_sharpe,
            "max_sharpe_drop_pct": round(max_drop, 4),
            "is_overfit": is_overfit,
            "reason": (
                "Sharpe dropped more than 50%; strategy may be overfit"
                if is_overfit
                else "Parameter sensitivity within acceptable range"
            ),
            "variants": variant_metrics,
        }
