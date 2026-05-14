"""
ATLAS Execution Engine
=======================
Routes approved orders to paper trading (Alpaca) or live exchange.
Monitors fills and updates the orders / positions tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config.settings import settings
from database import connection as db
from strategy_engine.base_strategy import TradeSignal, Signal


class ExecutionEngine:
    """Routes orders to Alpaca paper/live or Binance."""

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self._alpaca = None   # initialised lazily in setup()

    async def setup(self) -> None:
        """Import and initialise the Alpaca client."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce

            self._alpaca = TradingClient(
                api_key=settings.alpaca_api_key.get_secret_value(),
                secret_key=settings.alpaca_secret_key.get_secret_value(),
                paper=self.paper,
            )
            logger.info("[execution] Alpaca client ready (paper={})", self.paper)
        except ImportError:
            logger.warning("[execution] alpaca-py not installed — using mock mode.")

    async def submit_order(
        self,
        signal:   TradeSignal,
        quantity: float,
        strategy_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """
        Submit a market order for the given signal.
        Returns the raw exchange response dict.
        """
        if self._alpaca is None:
            return await self._mock_order(signal, quantity, strategy_id)

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if signal.signal == Signal.BUY else OrderSide.SELL
        req  = MarketOrderRequest(
            symbol       = signal.symbol,
            qty          = quantity,
            side         = side,
            time_in_force= TimeInForce.GTC,
        )
        try:
            order = self._alpaca.submit_order(req)
            raw   = order.model_dump() if hasattr(order, "model_dump") else dict(order)
            await self._persist_order(signal, quantity, strategy_id, raw)
            logger.info("[execution] Order submitted: {} {} {}", side, signal.symbol, quantity)
            return raw
        except Exception as exc:
            logger.error("[execution] Order submission failed: {}", exc)
            raise

    # ── Internals ─────────────────────────────────────────────

    async def _mock_order(
        self,
        signal:      TradeSignal,
        quantity:    float,
        strategy_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        """Simulate order fill for testing without a live connection."""
        raw = {
            "id":     str(uuid.uuid4()),
            "symbol": signal.symbol,
            "qty":    quantity,
            "side":   signal.signal.value.lower(),
            "status": "filled",
            "filled_avg_price": signal.entry_price,
            "mock": True,
        }
        await self._persist_order(signal, quantity, strategy_id, raw)
        logger.info("[execution] Mock order filled: {} {} @ {}", signal.signal, signal.symbol, signal.entry_price)
        return raw

    async def _persist_order(
        self,
        signal:      TradeSignal,
        quantity:    float,
        strategy_id: uuid.UUID | None,
        raw:         dict,
    ) -> None:
        await db.execute(
            """
            INSERT INTO orders
                (client_order_id, strategy_id, symbol, exchange,
                 side, order_type, status, quantity, limit_price,
                 stop_price, is_paper, raw_response)
            VALUES ($1,$2,$3,$4,$5,'market',$6,$7,$8,$9,$10,$11::jsonb)
            """,
            raw.get("id"),
            strategy_id,
            signal.symbol,
            "alpaca",
            signal.signal.value.lower(),
            raw.get("status", "submitted"),
            quantity,
            signal.entry_price,
            signal.stop_loss,
            self.paper,
            str(raw).replace("'", '"'),
        )
