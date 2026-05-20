"""
ATLAS E2E Pipeline Agent — agents/pipeline_agent.py
=====================================================
Runs every 60 seconds:
  1. Fetch latest features from feature_store
  2. Evaluate all active strategies
  3. Risk check each signal
  4. Submit paper trade via Alpaca
  5. Log everything to DB
  6. Zero manual steps required
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from agents.base_agent import BaseAgent
from database import connection as db
from risk_management.kill_switch import get_kill_switch
from risk_management.risk_manager import RiskManager


class PipelineAgent(BaseAgent):
    """Full E2E automated trading pipeline."""

    agent_type = "pipeline"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(
            name="pipeline_agent",
            config={"tick_seconds": 60, **(config or {})},
        )
        self._risk_manager: RiskManager | None = None
        self._executor = None
        self._cycle = 0

    async def setup(self) -> None:
        from execution.alpaca_executor import AlpacaExecutor
        self._executor = AlpacaExecutor()
        await self._executor.setup()
        self._risk_manager = RiskManager(capital=100_000.0, max_risk_per_trade=0.02)
        ks = get_kill_switch()
        await ks.setup()
        logger.info("[pipeline] E2E Pipeline Agent ready")

    async def run(self) -> None:
        self._cycle += 1
        logger.info("[pipeline] ── Cycle {} ──", self._cycle)

        # 1. Check kill switch
        ks = get_kill_switch()
        if ks.is_armed:
            logger.warning("[pipeline] Kill switch ARMED — skipping cycle")
            return

        # 2. Load active strategies
        strategies = await self._load_active_strategies()
        if not strategies:
            logger.warning("[pipeline] No active strategies found")
            return

        logger.info("[pipeline] {} active strategies", len(strategies))

        # 3. For each strategy, evaluate signals
        signals_generated = 0
        orders_submitted = 0

        for strategy in strategies:
            try:
                signals = await self._evaluate_strategy(strategy)
                signals_generated += len(signals)

                for signal in signals:
                    submitted = await self._process_signal(signal, strategy)
                    if submitted:
                        orders_submitted += 1

            except Exception as exc:
                logger.error("[pipeline] Strategy {} error: {}", strategy["name"], exc)
                continue

        logger.info(
            "[pipeline] Cycle {} done | strategies={} signals={} orders={}",
            self._cycle, len(strategies), signals_generated, orders_submitted,
        )

        # 4. Log cycle summary to agent_logs
        await self._log_cycle(len(strategies), signals_generated, orders_submitted)

    async def teardown(self) -> None:
        if self._executor:
            await self._executor.teardown()
        logger.info("[pipeline] Pipeline Agent stopped")

    # ── Internal methods ──────────────────────────────────────────────────────

    async def _load_active_strategies(self) -> list[dict]:
        """Load all active strategies from DB."""
        try:
            rows = await db.fetch(
                """SELECT id, name, strategy_type, symbols, parameters, code
                   FROM strategies
                   WHERE status = 'active'
                   LIMIT 10"""
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("[pipeline] Failed to load strategies: {}", exc)
            return []

    async def _evaluate_strategy(self, strategy: dict) -> list[dict]:
        """Fetch features and generate signals for a strategy."""
        signals = []
        symbols = strategy.get("symbols") or []
        if isinstance(symbols, str):
            import json
            symbols = json.loads(symbols)

        for symbol in symbols:
            try:
                features = await self._fetch_features(symbol)
                if features is None or features.empty:
                    logger.debug("[pipeline] No features for {}", symbol)
                    continue

                signal = self._compute_signal(symbol, features, strategy)
                if signal:
                    signals.append(signal)

            except Exception as exc:
                logger.error("[pipeline] Signal error for {}: {}", symbol, exc)

        return signals

    async def _fetch_features(self, symbol: str) -> pd.DataFrame | None:
        """Get latest features for a symbol from feature_store."""
        try:
            rows = await db.fetch(
                """SELECT feature_name, feature_value
                   FROM feature_store
                   WHERE symbol = $1
                   AND computed_at > NOW() - INTERVAL '5 minutes'
                   ORDER BY computed_at DESC""",
                symbol,
            )
            if not rows:
                return None

            data = {row["feature_name"]: row["feature_value"] for row in rows}
            return pd.DataFrame([data])

        except Exception as exc:
            logger.error("[pipeline] Feature fetch error for {}: {}", symbol, exc)
            return None

    def _compute_signal(self, symbol: str, features: pd.DataFrame, strategy: dict) -> dict | None:
        """
        Simple signal logic using features.
        Returns BUY/SELL/HOLD based on EMA crossover + RSI.
        """
        try:
            row = features.iloc[0]

            ema_9  = row.get("ema_9",  0)
            ema_21 = row.get("ema_21", 0)
            rsi_14 = row.get("rsi_14", 50)
            regime = row.get("regime_score", 0)

            # No signal if missing data
            if not ema_9 or not ema_21:
                return None

            side = None

            # BUY: EMA9 > EMA21, RSI not overbought, bullish regime
            if ema_9 > ema_21 and rsi_14 < 70 and regime >= 0:
                side = "buy"

            # SELL: EMA9 < EMA21, RSI not oversold
            elif ema_9 < ema_21 and rsi_14 > 30:
                side = "sell"

            if not side:
                return None

            # Estimate stop loss using ATR
            atr = row.get("atr_14", 1.0) or 1.0

            return {
                "symbol":      symbol,
                "side":        side,
                "strategy_id": str(strategy["id"]),
                "strategy_name": strategy["name"],
                "confidence":  0.7,
                "stop_loss":   atr * 2,
                "qty":         1.0,
            }

        except Exception as exc:
            logger.error("[pipeline] Compute signal error: {}", exc)
            return None

    async def _process_signal(self, signal: dict, strategy: dict) -> bool:
        """Risk check and submit order for a signal."""
        symbol = signal["symbol"]
        side   = signal["side"]

        try:
            # Risk check
            if not self._risk_manager:
                return False

            from strategy_engine.base_strategy import TradeSignal, Signal
            ts = TradeSignal(
                symbol=symbol,
                signal=Signal.BUY if side == "buy" else Signal.SELL,
                strategy_id=signal["strategy_id"],
                strategy_name=signal["strategy_name"],
                confidence=signal["confidence"],
                stop_loss=signal.get("stop_loss", 1.0),
                take_profit=signal.get("stop_loss", 1.0) * 2,
            )

            # Get current price from features
            price_row = await db.fetchrow(
                """SELECT close FROM market_data
                   WHERE symbol = $1
                   ORDER BY timestamp DESC LIMIT 1""",
                symbol,
            )
            current_price = float(price_row["close"]) if price_row else 100.0

            result = self._risk_manager.check_signal(ts, current_price=current_price)
            if not result.approved:
                logger.debug("[pipeline] Signal rejected: {} — {}", symbol, result.rejection_reason)
                return False

            # Submit paper order
            order = await self._executor.submit_order(
                symbol=symbol,
                qty=signal["qty"],
                side=side,
                order_type="market",
                strategy_id=signal["strategy_id"],
                stop_loss=current_price - signal.get("stop_loss", 1.0),
            )

            logger.success(
                "[pipeline] ✅ Order submitted | {} {} 1 share @ ~${:.2f} | id={}",
                side.upper(), symbol, current_price,
                str(order.get("id", ""))[:8],
            )
            return True

        except Exception as exc:
            logger.error("[pipeline] Order error for {}: {}", symbol, exc)
            return False

    async def _log_cycle(self, strategies: int, signals: int, orders: int) -> None:
        """Log pipeline cycle summary to agent_logs."""
        try:
            await db.execute(
                """INSERT INTO agent_logs (agent_name, level, message, created_at)
                   VALUES ($1, $2, $3, NOW())
                   ON CONFLICT DO NOTHING""",
                "pipeline_agent",
                "INFO",
                f"Cycle {self._cycle} | strategies={strategies} signals={signals} orders={orders}",
            )
        except Exception:
            pass
