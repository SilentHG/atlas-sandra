"""
ATLAS — Binance WebSocket Kline Collector
==========================================
Streams real-time 1-minute kline data for crypto pairs from Binance
and persists CLOSED bars to the TimescaleDB `market_data` hypertable.

Supported pairs (Day 1): BTCUSDT, ETHUSDT, SOLUSDT

Uses the Binance combined-stream endpoint (no API key required for
public market data). Only finalised bars (kline.x == True) are saved.

Reconnection: tenacity exponential back-off (1s → 2s → 4s … max 60s).

Run standalone:
    python -m data_ingestion.binance_collector
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

# ── Constants ──────────────────────────────────────────────────────────────────

BINANCE_WS_BASE  = "wss://stream.binance.com:9443/stream"
CRYPTO_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
KLINE_INTERVAL   = "1m"

_STREAMS         = "/".join(f"{s.lower()}@kline_{KLINE_INTERVAL}" for s in CRYPTO_SYMBOLS)
BINANCE_WS_URL   = f"{BINANCE_WS_BASE}?streams={_STREAMS}"

_BACKOFF = wait_exponential(multiplier=1, min=1, max=60)


# ── Parser ─────────────────────────────────────────────────────────────────────


def _parse_kline(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Return a normalised bar dict from a Binance combined-stream kline event,
    or None if the bar is not yet closed or the payload is not a kline event.

    Binance combined-stream envelope:
        { "stream": "btcusdt@kline_1m", "data": { "e": "kline", ... } }

    Key kline fields:
        k.t  – open  time (ms)   k.o / k.h / k.l / k.c – OHLC
        k.v  – base volume       k.n – number of trades
        k.x  – is kline closed (bool)
    """
    try:
        data = payload.get("data", payload)
        if data.get("e") != "kline":
            return None
        k = data["k"]
        if not k.get("x", False):
            return None   # interim update — skip

        return {
            "symbol":     k["s"],
            "timestamp":  datetime.fromtimestamp(k["t"] / 1_000, tz=timezone.utc),
            "open":       float(k["o"]),
            "high":       float(k["h"]),
            "low":        float(k["l"]),
            "close":      float(k["c"]),
            "volume":     float(k["v"]),
            "vwap":       None,
            "num_trades": int(k["n"]),
            "exchange":   "binance",
            "source":     "binance",
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("[binance] Parse error: {} | payload={}", exc, payload)
        return None


# ── DB writer ──────────────────────────────────────────────────────────────────


async def _persist_bar(bar: dict[str, Any]) -> None:
    """Upsert one closed kline bar into market_data."""
    await db.execute(
        """
        INSERT INTO market_data
            (symbol, timestamp, open, high, low, close,
             volume, vwap, num_trades, exchange, source)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        ON CONFLICT (symbol, timestamp) DO UPDATE
            SET open        = EXCLUDED.open,
                high        = EXCLUDED.high,
                low         = EXCLUDED.low,
                close       = EXCLUDED.close,
                volume      = EXCLUDED.volume,
                num_trades  = EXCLUDED.num_trades,
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
        "[binance] ✓ {} | {:%Y-%m-%d %H:%M} UTC | "
        "O={:.4f} H={:.4f} L={:.4f} C={:.4f} V={:.4f}",
        bar["symbol"], bar["timestamp"],
        bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"],
    )


# ── WebSocket session ──────────────────────────────────────────────────────────


async def _run_session() -> None:
    """Single WebSocket connection lifecycle. Raises on error → triggers retry."""
    logger.info("[binance] Connecting to {}", BINANCE_WS_URL)

    async with websockets.connect(
        BINANCE_WS_URL,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
        max_size=10 * 1024 * 1024,
    ) as ws:
        logger.success(
            "[binance] Connected — streaming {} @ {}",
            CRYPTO_SYMBOLS, KLINE_INTERVAL,
        )

        async for raw in ws:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("[binance] JSON decode error: {}", exc)
                continue

            bar = _parse_kline(payload)
            if bar is None:
                continue   # interim update or non-kline event

            try:
                await _persist_bar(bar)
            except Exception as exc:
                logger.error("[binance] DB write error for {}: {}", bar.get("symbol"), exc)


# ── Public runner ──────────────────────────────────────────────────────────────


async def run_binance_collector() -> None:
    """Start the Binance collector with unlimited exponential back-off reconnects."""
    logger.info("[binance] Starting collector for {}", CRYPTO_SYMBOLS)
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
            await _run_session()

    logger.warning("[binance] Collector exited retry loop unexpectedly")


# ── CLI ────────────────────────────────────────────────────────────────────────


async def _main() -> None:
    loop      = asyncio.get_running_loop()
    stop_evt  = asyncio.Event()

    def _shutdown(sig_name: str) -> None:
        logger.info("[binance] {} — shutting down …", sig_name)
        stop_evt.set()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig.name)

    task = asyncio.create_task(run_binance_collector())
    try:
        await asyncio.wait(
            [task, asyncio.create_task(stop_evt.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await db.close_pool()
        logger.info("[binance] Collector stopped.")


if __name__ == "__main__":
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
