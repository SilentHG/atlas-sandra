"""
ATLAS — Polygon.io WebSocket Collector
=======================================
Streams real-time 1-minute OHLCV bars for US equities from Polygon.io
and persists them to the TimescaleDB `market_data` hypertable.

Supported symbols (Day 1): AAPL, MSFT, NVDA, TSLA, AMZN

WebSocket docs:
  https://polygon.io/docs/stocks/ws_stocks_am

Message flow:
  1. Connect to wss://socket.polygon.io/stocks
  2. Authenticate with API key
  3. Subscribe to AM.* (per-minute OHLCV aggregates)
  4. On each AM message → validate → INSERT market_data
  5. On any error → exponential back-off → reconnect

Run standalone:
  python -m data_ingestion.polygon_collector
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Any

import websockets
from loguru import logger
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_never,
    wait_exponential,
)

from config.settings import settings
from database import connection as db

# ── Constants ─────────────────────────────────────────────────────────────────

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

EQUITY_SYMBOLS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Subscription string: AM = Aggregate-per-Minute bars
SUBSCRIBE_CHANNELS = [f"AM.{sym}" for sym in EQUITY_SYMBOLS]

# Reconnect back-off: 1s → 2s → 4s … capped at 60s
_BACKOFF = wait_exponential(multiplier=1, min=1, max=60)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_am_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Parse a Polygon AM (aggregate-per-minute) WebSocket message into a
    normalised bar dict ready for DB insertion.

    Polygon AM fields:
      ev  – event type ("AM")
      sym – symbol
      v   – volume
      av  – accumulated volume
      op  – official open price
      vw  – volume-weighted average price
      o   – open
      c   – close
      h   – high
      l   – low
      a   – VWAP
      z   – average trade size
      s   – start timestamp (ms)
      e   – end timestamp   (ms)
      n   – number of trades in window
      x   – exchange id (integer)
    """
    if msg.get("ev") != "AM":
        return None
    try:
        ts = datetime.fromtimestamp(msg["s"] / 1_000, tz=timezone.utc)
        return {
            "symbol":     msg["sym"],
            "timestamp":  ts,
            "open":       float(msg["o"]),
            "high":       float(msg["h"]),
            "low":        float(msg["l"]),
            "close":      float(msg["c"]),
            "volume":     float(msg["v"]),
            "vwap":       float(msg.get("vw") or msg.get("a") or 0),
            "num_trades": int(msg.get("n", 0)),
            "exchange":   str(msg.get("x", "polygon")),
            "source":     "polygon",
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("[polygon] Failed to parse AM message: {} | msg={}", exc, msg)
        return None


async def _persist_bar(bar: dict[str, Any]) -> None:
    """Upsert a single OHLCV bar into market_data."""
    await db.execute(
        """
        INSERT INTO market_data
            (symbol, timestamp, open, high, low, close,
             volume, vwap, num_trades, exchange, source)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT (symbol, timestamp) DO UPDATE
            SET open       = EXCLUDED.open,
                high       = EXCLUDED.high,
                low        = EXCLUDED.low,
                close      = EXCLUDED.close,
                volume     = EXCLUDED.volume,
                vwap       = EXCLUDED.vwap,
                num_trades = EXCLUDED.num_trades,
                ingested_at = NOW()
        """,
        bar["symbol"],
        bar["timestamp"],
        bar["open"],
        bar["high"],
        bar["low"],
        bar["close"],
        bar["volume"],
        bar["vwap"],
        bar["num_trades"],
        bar["exchange"],
        bar["source"],
    )
    logger.info(
        "[polygon] ✓ {} | {:%Y-%m-%d %H:%M} UTC | O={:.4f} H={:.4f} L={:.4f} C={:.4f} V={:.0f}",
        bar["symbol"],
        bar["timestamp"],
        bar["open"],
        bar["high"],
        bar["low"],
        bar["close"],
        bar["volume"],
    )


# ── Core WebSocket session ────────────────────────────────────────────────────


async def _run_session(api_key: str) -> None:
    """
    Open ONE WebSocket session to Polygon, authenticate, subscribe,
    and process messages until the connection drops or an auth error occurs.
    Raises on error so the retry decorator can handle reconnection.
    """
    logger.info("[polygon] Connecting to {}", POLYGON_WS_URL)

    async with websockets.connect(
        POLYGON_WS_URL,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
        max_size=10 * 1024 * 1024,   # 10 MB – avoids frame-size errors
    ) as ws:
        # ── Step 1: receive connection banner ────────────────────────────────
        raw = await ws.recv()
        msgs = json.loads(raw)
        for m in msgs if isinstance(msgs, list) else [msgs]:
            logger.debug("[polygon] banner: {}", m)
            if m.get("status") == "connected":
                logger.success("[polygon] Connected to Polygon.io WebSocket")

        # ── Step 2: authenticate ─────────────────────────────────────────────
        await ws.send(json.dumps({"action": "auth", "params": api_key}))
        raw = await ws.recv()
        msgs = json.loads(raw)
        for m in msgs if isinstance(msgs, list) else [msgs]:
            status = m.get("status", "")
            if status == "auth_success":
                logger.success("[polygon] Authentication successful")
            elif status in ("auth_failed", "auth_timeout"):
                logger.error("[polygon] Authentication failed: {}", m)
                raise PermissionError(f"Polygon auth failed: {m}")
            else:
                logger.debug("[polygon] auth response: {}", m)

        # ── Step 3: subscribe ────────────────────────────────────────────────
        subscribe_msg = json.dumps(
            {"action": "subscribe", "params": ",".join(SUBSCRIBE_CHANNELS)}
        )
        await ws.send(subscribe_msg)
        logger.info("[polygon] Subscribed to: {}", SUBSCRIBE_CHANNELS)

        # ── Step 4: consume messages ─────────────────────────────────────────
        async for raw_msg in ws:
            try:
                messages = json.loads(raw_msg)
            except json.JSONDecodeError as exc:
                logger.warning("[polygon] JSON decode error: {}", exc)
                continue

            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                ev = msg.get("ev")

                if ev == "AM":
                    bar = _parse_am_message(msg)
                    if bar:
                        try:
                            await _persist_bar(bar)
                        except Exception as db_exc:
                            logger.error("[polygon] DB write error: {}", db_exc)

                elif ev == "status":
                    logger.debug("[polygon] status: {}", msg)

                else:
                    logger.trace("[polygon] unhandled ev='{}': {}", ev, msg)


# ── Public entry point ────────────────────────────────────────────────────────


async def run_polygon_collector() -> None:
    """
    Start the Polygon collector with automatic reconnection.
    Uses tenacity for exponential back-off on any connection/protocol error.
    A PermissionError (bad API key) stops retrying immediately.
    """
    api_key = settings.polygon_api_key.get_secret_value()
    logger.info(
        "[polygon] Starting collector for symbols: {}", EQUITY_SYMBOLS
    )

    # Initialise DB pool once
    await db.init_pool()

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(
            (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
                asyncio.TimeoutError,
            )
        ),
        wait=_BACKOFF,
        stop=stop_never,
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=False,
    ):
        with attempt:
            try:
                await _run_session(api_key)
            except PermissionError:
                logger.critical(
                    "[polygon] Invalid API key — stopping collector. "
                    "Check POLYGON_API_KEY in config/keys.env"
                )
                raise  # bubble up — do not retry

    # Should never reach here (stop=stop_never), but log just in case
    logger.warning("[polygon] Collector exited retry loop unexpectedly")


# ── CLI entrypoint ────────────────────────────────────────────────────────────


async def _main() -> None:
    """Stand-alone runner with graceful shutdown on SIGINT / SIGTERM."""

    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _shutdown(sig_name: str) -> None:
        logger.info("[polygon] {} received — shutting down …", sig_name)
        stop_event.set()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig.name)

    collector_task = asyncio.create_task(run_polygon_collector())

    try:
        await asyncio.wait(
            [collector_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        collector_task.cancel()
        try:
            await collector_task
        except asyncio.CancelledError:
            pass
        await db.close_pool()
        logger.info("[polygon] Collector stopped.")


if __name__ == "__main__":
    # Configure loguru for stdout
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
    asyncio.run(_main())
