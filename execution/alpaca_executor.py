"""
ATLAS Alpaca Executor — execution/alpaca_executor.py
=====================================================
Connects to Alpaca paper trading API, submits orders with full
pre-trade risk checks (kill switch + risk manager), persists
orders and positions to DB.

EXEC-001: Submit orders to https://paper-api.alpaca.markets
EXEC-002: Check kill switch + risk limits before every order
"""
from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Any
import httpx
from loguru import logger
from config.settings import settings
from database import connection as db
from risk_management.kill_switch import get_kill_switch
from risk_management.risk_manager import RiskManager

_BASE = settings.alpaca_base_url.rstrip("/")

# Singleton risk manager for pre-trade checks
_risk_manager: RiskManager | None = None


def _get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(capital=100_000.0)
    return _risk_manager


class AlpacaExecutor:
    def __init__(self) -> None:
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None
        self._ks = get_kill_switch()

    async def setup(self) -> None:
        try:
            self._client = httpx.AsyncClient(headers=self._headers, timeout=10.0)
            logger.info("[alpaca_exec] ready — {}", _BASE)
        except Exception as exc:
            logger.error("[alpaca_exec] setup failed: {}", exc)
            raise

    async def teardown(self) -> None:
        try:
            if self._client:
                await self._client.aclose()
        except Exception as exc:
            logger.error("[alpaca_exec] teardown error: {}", exc)

    async def get_account(self) -> dict:
        try:
            r = await self._client.get(f"{_BASE}/v2/account")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.error("[alpaca_exec] get_account error: {}", exc)
            raise

    async def get_positions(self) -> list:
        try:
            r = await self._client.get(f"{_BASE}/v2/positions")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.error("[alpaca_exec] get_positions error: {}", exc)
            raise

    async def cancel_all_orders(self) -> int:
        try:
            r = await self._client.delete(f"{_BASE}/v2/orders")
            data = r.json() if r.text else []
            n = len(data) if isinstance(data, list) else 0
            logger.warning("[alpaca_exec] Cancelled {} orders", n)
            return n
        except Exception as e:
            logger.error("[alpaca_exec] Cancel failed: {}", e)
            return 0

    async def test_connectivity(self) -> dict:
        """
        Submit a test market order for AAPL 1 share to verify connectivity.
        Returns fill confirmation with timing.
        """
        try:
            t0 = time.perf_counter()
            result = await self.submit_order(
                symbol="AAPL", qty=1, side="buy",
                order_type="market", strategy_id=None, stop_loss=None,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("[alpaca_exec] Test order completed in {:.0f}ms", elapsed_ms)
            return {
                "status": "ok",
                "order": result,
                "elapsed_ms": round(elapsed_ms, 1),
                "within_500ms": elapsed_ms < 500,
            }
        except Exception as exc:
            logger.error("[alpaca_exec] Test connectivity failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        strategy_id: str | None = None,
        stop_loss: float | None = None,
    ) -> dict:
        """Submit order with full pre-trade risk checks."""
        t0 = time.perf_counter()

        # ── EXEC-002: Check kill switch BEFORE order ──────────────────────────
        try:
            if await self._ks.is_armed():
                s = await self._ks.get_global_state()
                reason = f"Kill switch active: {s.get('reason')}"
                logger.critical("[alpaca_exec] ORDER BLOCKED: {}", reason)
                raise RuntimeError(reason)
            if strategy_id and await self._ks.is_strategy_killed(strategy_id):
                reason = f"Strategy {strategy_id} killed"
                logger.warning("[alpaca_exec] ORDER BLOCKED: {}", reason)
                raise RuntimeError(reason)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("[alpaca_exec] Kill switch check error: {}", exc)

        # ── EXEC-002: Check risk limits BEFORE order ──────────────────────────
        try:
            rm = _get_risk_manager()
            # Get current price for risk check
            price_row = await db.fetchval(
                "SELECT close FROM market_data WHERE symbol=$1 ORDER BY timestamp DESC LIMIT 1",
                symbol,
            )
            current_price = float(price_row) if price_row else 0.0

            if current_price > 0:
                risk_result = await rm.validate_order(
                    symbol=symbol, qty=qty, side=side,
                    price=current_price, strategy_id=strategy_id,
                )
                if not risk_result.approved:
                    logger.warning(
                        "[alpaca_exec] ORDER REJECTED by risk manager: {} — {}",
                        symbol, risk_result.rejection_reason,
                    )
                    raise RuntimeError(f"Risk check failed: {risk_result.rejection_reason}")
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[alpaca_exec] Risk check error (proceeding): {}", exc)

        # ── Submit to Alpaca ──────────────────────────────────────────────────
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": "gtc",
            "client_order_id": str(uuid.uuid4()),
        }
        if order_type == "limit" and limit_price:
            payload["limit_price"] = str(limit_price)

        try:
            r = await self._client.post(f"{_BASE}/v2/orders", content=json.dumps(payload))
            r.raise_for_status()
            raw = r.json()
        except httpx.HTTPStatusError as e:
            logger.error("[alpaca_exec] Alpaca HTTP error: {}", e.response.text)
            raise RuntimeError(f"Alpaca: {e.response.text}") from e
        except Exception as exc:
            logger.error("[alpaca_exec] Order submission error: {}", exc)
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[alpaca_exec] ✓ {} {} {} {} | {:.0f}ms",
            order_type.upper(), side.upper(), qty, symbol, elapsed_ms,
        )

        # ── Persist order and sync positions ──────────────────────────────────
        asyncio.create_task(self._persist(raw, strategy_id, stop_loss))
        asyncio.create_task(self.sync_positions())

        return raw

    async def _persist(self, raw: dict, strategy_id: str | None, stop_loss: float | None) -> None:
        """Save order to orders table."""
        try:
            await db.execute(
                """INSERT INTO orders
                   (client_order_id,strategy_id,symbol,exchange,side,order_type,
                    status,quantity,limit_price,stop_price,is_paper,raw_response,submitted_at)
                   VALUES ($1,$2,$3,'alpaca',$4,$5,$6,$7,$8,$9,TRUE,$10::jsonb,NOW())
                   ON CONFLICT (client_order_id) DO NOTHING""",
                raw.get("client_order_id"),
                uuid.UUID(strategy_id) if strategy_id else None,
                raw.get("symbol"),
                raw.get("side"),
                raw.get("type"),
                raw.get("status", "submitted"),
                float(raw.get("qty", 0)),
                float(raw.get("limit_price") or 0) or None,
                stop_loss,
                json.dumps(raw),
            )
            logger.debug("[alpaca_exec] Order persisted: {}", raw.get("client_order_id"))
        except Exception as e:
            logger.error("[alpaca_exec] persist order failed: {}", e)

    async def sync_positions(self) -> None:
        """Sync positions from Alpaca to positions table."""
        try:
            positions = await self.get_positions()
            for p in positions:
                await db.execute(
                    """INSERT INTO positions
                       (symbol, side, quantity, entry_price, current_price,
                        unrealized_pnl, pnl, status, is_paper)
                       VALUES ($1, $2, $3, $4, $5, $6, $6, 'open', TRUE)
                       ON CONFLICT DO NOTHING""",
                    p.get("symbol"),
                    p.get("side", "long"),
                    float(p.get("qty", 0)),
                    float(p.get("avg_entry_price", 0)),
                    float(p.get("current_price", 0)),
                    float(p.get("unrealized_pl", 0)),
                )
            logger.debug("[alpaca_exec] Synced {} positions from Alpaca", len(positions))
        except Exception as e:
            logger.error("[alpaca_exec] sync_positions: {}", e)
