"""
ATLAS Execution Engine
=======================
Routes approved orders to paper trading (Alpaca) or live exchange.
Monitors fills and updates the orders / positions tables.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from database import connection as db
from strategy_engine.base_strategy import TradeSignal, Signal


class ExecutionEngine:
    """Routes orders to Alpaca paper/live or Binance."""

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self._alpaca = None   # initialised lazily in setup()

    async def setup(self) -> None:
        """Initialise the Alpaca paper trading client."""
        try:
            from data_ingestion.alpaca_connector import AlpacaConnector

            self._alpaca = AlpacaConnector()
            await self._alpaca._open()
            logger.info("[execution] Alpaca client ready (paper={})", self.paper)
        except Exception as exc:
            logger.error("[execution] Alpaca setup failed: {}", exc)
            raise

    async def teardown(self) -> None:
        if self._alpaca is not None:
            await self._alpaca._close()

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
            raise RuntimeError("ExecutionEngine.setup() must complete before submitting orders.")

        side = "buy" if signal.signal == Signal.BUY else "sell"
        try:
            raw = await self._alpaca.submit_order(
                symbol=signal.symbol,
                qty=quantity,
                side=side,
                order_type="market",
                time_in_force="day",
                client_order_id=str(uuid.uuid4()),
            )
            await self._persist_order(signal, quantity, strategy_id, raw)
            logger.info("[execution] Order submitted: {} {} {}", side, signal.symbol, quantity)
            return raw
        except Exception as exc:
            logger.error("[execution] Order submission failed: {}", exc)
            raise

    # ── Internals ─────────────────────────────────────────────

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
            json.dumps(raw),
        )
