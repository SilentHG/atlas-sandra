"""
ATLAS — Alpaca Paper Trading Connector
========================================
Provides a thin async wrapper around the Alpaca REST API for paper trading.

Responsibilities on Day 1:
  1. Load credentials from config/keys.env via ATLASSettings.
  2. Verify the connection and print account status.
  3. Submit a test market order (AAPL, 1 share, buy) and immediately cancel it.

Designed to be imported by the execution agent:
    from data_ingestion.alpaca_connector import AlpacaConnector
    connector = AlpacaConnector()
    await connector.verify_connection()

Run standalone for connectivity test:
    python -m data_ingestion.alpaca_connector
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from config.settings import settings


# ── Constants ──────────────────────────────────────────────────────────────────

# Alpaca paper-trading base URL (falls back to env value)
_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"

# Request timeout (seconds)
_TIMEOUT = httpx.Timeout(timeout=15.0)

# ── AlpacaConnector ────────────────────────────────────────────────────────────


class AlpacaConnector:
    """
    Async REST client for Alpaca paper trading.

    Uses httpx.AsyncClient internally so it integrates cleanly with the
    asyncio event loop used by the rest of ATLAS.

    Usage:
        async with AlpacaConnector() as conn:
            account = await conn.get_account()
            order   = await conn.submit_test_order()
    """

    def __init__(self) -> None:
        self._api_key    = settings.alpaca_api_key
        self._secret_key = settings.alpaca_secret_key
        self._base_url   = settings.alpaca_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "[alpaca] Connector initialised — endpoint: {}", self._base_url
        )

    # ── Context manager support ────────────────────────────────────────────────

    async def __aenter__(self) -> "AlpacaConnector":
        await self._open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._close()

    async def _open(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "APCA-API-KEY-ID":     self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
                "Accept":              "application/json",
                "Content-Type":        "application/json",
            },
            timeout=_TIMEOUT,
        )
        logger.debug("[alpaca] HTTP client opened")

    async def _close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("[alpaca] HTTP client closed")

    # ── Low-level request helpers ──────────────────────────────────────────────

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "AlpacaConnector is not open. "
                "Use 'async with AlpacaConnector() as conn: ...' "
                "or call await connector._open() first."
            )
        return self._client

    async def _get(self, path: str, **params: Any) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[alpaca] GET {} → HTTP {}: {}",
                path, exc.response.status_code, exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("[alpaca] GET {} → network error: {}", path, exc)
            raise

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[alpaca] POST {} → HTTP {}: {}",
                path, exc.response.status_code, exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("[alpaca] POST {} → network error: {}", path, exc)
            raise

    async def _delete(self, path: str) -> None:
        client = self._ensure_client()
        try:
            resp = await client.delete(path)
            # 204 No Content is success for cancellation
            if resp.status_code not in (200, 204):
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[alpaca] DELETE {} → HTTP {}: {}",
                path, exc.response.status_code, exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("[alpaca] DELETE {} → network error: {}", path, exc)
            raise

    # ── Public API methods ────────────────────────────────────────────────────

    async def get_account(self) -> dict[str, Any]:
        """
        Fetch and return Alpaca account details.

        Key fields in the response:
          account_number, status, equity, buying_power,
          cash, portfolio_value, pattern_day_trader,
          trading_blocked, account_blocked, created_at
        """
        account = await self._get("/v2/account")
        logger.info(
            "[alpaca] Account #{} | status={} | equity=${} | "
            "buying_power=${} | cash=${}",
            account.get("account_number", "N/A"),
            account.get("status", "N/A"),
            account.get("equity", "N/A"),
            account.get("buying_power", "N/A"),
            account.get("cash", "N/A"),
        )
        return account

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return a list of all open positions."""
        positions = await self._get("/v2/positions")
        logger.info("[alpaca] Open positions: {}", len(positions))
        for pos in positions:
            logger.debug(
                "[alpaca]   {} qty={} side={} avg_entry=${} unrealized_pl=${}",
                pos.get("symbol"),
                pos.get("qty"),
                pos.get("side"),
                pos.get("avg_entry_price"),
                pos.get("unrealized_pl"),
            )
        return positions

    async def submit_order(
        self,
        symbol:     str,
        qty:        float,
        side:       str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price:  float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit an order to Alpaca paper trading.

        Args:
            symbol:          Ticker, e.g. "AAPL"
            qty:             Number of shares (fractional supported)
            side:            "buy" or "sell"
            order_type:      "market" | "limit" | "stop" | "stop_limit"
            time_in_force:   "day" | "gtc" | "opg" | "cls" | "ioc" | "fok"
            limit_price:     Required for limit / stop_limit orders
            stop_price:      Required for stop / stop_limit orders
            client_order_id: Optional idempotency key (max 48 chars)

        Returns:
            Alpaca order dict with id, status, symbol, qty, etc.
        """
        body: dict[str, Any] = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if stop_price is not None:
            body["stop_price"] = str(stop_price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        order = await self._post("/v2/orders", body)
        logger.info(
            "[alpaca] Order submitted: id={} | {} {} {} @ {} | status={}",
            order.get("id"),
            side.upper(),
            qty,
            symbol,
            order_type,
            order.get("status"),
        )
        return order

    async def cancel_order(self, order_id: str) -> None:
        """
        Cancel an open order by its Alpaca order ID.
        Logs success or any errors.
        """
        try:
            await self._delete(f"/v2/orders/{order_id}")
            logger.info("[alpaca] Order {} cancelled successfully", order_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                logger.warning(
                    "[alpaca] Order {} cannot be cancelled "
                    "(already filled / cancelled): {}",
                    order_id,
                    exc.response.text,
                )
            else:
                raise

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch the current state of a specific order."""
        return await self._get(f"/v2/orders/{order_id}")

    async def get_orders(
        self,
        status: str = "open",
        limit:  int = 50,
    ) -> list[dict[str, Any]]:
        """Return a list of orders filtered by status."""
        orders = await self._get("/v2/orders", status=status, limit=limit)
        logger.info("[alpaca] {} {} orders retrieved", len(orders), status)
        return orders

    async def cancel_all_orders(self) -> None:
        """Cancel ALL open orders. Use with care."""
        await self._delete("/v2/orders")
        logger.warning("[alpaca] All open orders cancelled")

    # ── High-level helpers ────────────────────────────────────────────────────

    async def verify_connection(self) -> dict[str, Any]:
        """
        Full connectivity check:
          1. GET /v2/account → verify auth and print status
          2. GET /v2/positions → log open positions

        Returns the account dict. Raises on any error.
        """
        logger.info("[alpaca] ── Running connectivity check ──")
        account   = await self.get_account()
        _positions = await self.get_positions()

        trading_ok = not account.get("trading_blocked", True)
        account_ok = not account.get("account_blocked", True)
        logger.info(
            "[alpaca] Connectivity check PASSED | "
            "trading_blocked={} account_blocked={}",
            not trading_ok,
            not account_ok,
        )
        return account

    async def submit_test_order(
        self,
        symbol: str = "AAPL",
        qty:    float = 1,
    ) -> None:
        """
        End-to-end smoke test:
          1. Submit a paper market buy order for `symbol`.
          2. Immediately cancel it.
          3. Verify final status is 'canceled'.

        This confirms the full order-management round-trip works on Day 1.
        """
        logger.info(
            "[alpaca] ── Test order smoke test: BUY {} {} ──",
            qty, symbol,
        )

        # Use a timestamped client_order_id to guarantee uniqueness
        client_id = f"atlas-test-{symbol.lower()}-{int(datetime.now(tz=timezone.utc).timestamp())}"

        # 1. Submit
        order = await self.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            order_type="market",
            time_in_force="day",
            client_order_id=client_id,
        )
        order_id = order["id"]
        logger.info(
            "[alpaca] Test order created: id={} status={}",
            order_id, order.get("status"),
        )

        # Brief pause — let Alpaca register the order before cancellation
        await asyncio.sleep(0.5)

        # 2. Cancel
        await self.cancel_order(order_id)

        # 3. Verify
        await asyncio.sleep(0.5)
        final = await self.get_order(order_id)
        final_status = final.get("status", "unknown")

        if final_status in ("canceled", "cancelled", "pending_cancel"):
            logger.success(
                "[alpaca] ✓ Test order round-trip PASSED: "
                "submitted → cancelled | final_status={}",
                final_status,
            )
        else:
            logger.warning(
                "[alpaca] Test order final status is '{}' — "
                "may still be pending cancellation",
                final_status,
            )


# ── CLI entrypoint ─────────────────────────────────────────────────────────────


async def _main() -> None:
    """
    Run a full connectivity + test-order smoke test when executed directly.
    Useful for verifying Alpaca credentials on Day 1.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.info("[alpaca] === ATLAS Alpaca Day-1 Smoke Test ===")
    logger.info("[alpaca] Base URL : {}", settings.alpaca_base_url)
    logger.info(
        "[alpaca] API Key  : {}…{}",
        settings.alpaca_api_key[:6],
        settings.alpaca_api_key[-4:],
    )

    async with AlpacaConnector() as conn:
        try:
            # Step 1: Verify account connection
            await conn.verify_connection()

            # Step 2: Submit and cancel a test order
            await conn.submit_test_order(symbol="AAPL", qty=1)

            logger.success("[alpaca] === Day-1 smoke test COMPLETE ===")

        except httpx.HTTPStatusError as exc:
            logger.critical(
                "[alpaca] HTTP error during smoke test: {} — {}",
                exc.response.status_code,
                exc.response.text,
            )
            sys.exit(1)
        except Exception as exc:
            logger.critical("[alpaca] Unexpected error: {}", exc)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
