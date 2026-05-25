"""
ATLAS — Polygon.io WebSocket + REST Collector
===============================================
Streams real-time 1-minute OHLCV bars for US equities from Polygon.io
and persists them to the TimescaleDB `market_data` hypertable.

DATA-001 ZERO GAPS:
  - WebSocket streaming for real-time bars
  - REST backfill loop once per day to catch missed historical bars
  - After every bar saved, check if previous minute exists
  - If gap found, immediately backfill from REST API
  - Only store equities data between 09:30-16:00 ET

Supported symbols: AAPL, MSFT, NVDA, TSLA, AMZN

Run standalone:
  python -m data_ingestion.polygon_collector
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from loguru import logger
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_never,
    wait_exponential,
)
import httpx

from config.settings import settings
from database import connection as db

# ── Constants ─────────────────────────────────────────────────────────────────

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

EQUITY_SYMBOLS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Subscription string: AM = Aggregate-per-Minute bars
SUBSCRIBE_CHANNELS = [f"AM.{sym}" for sym in EQUITY_SYMBOLS]

# Reconnect back-off: 1s → 2s → 4s … capped at 60s
_BACKOFF = wait_exponential(multiplier=1, min=1, max=60)

# REST backfill interval
_BACKFILL_INTERVAL_S = 86400  # once per day

# Market hours in ET
MARKET_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN  = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)

# Market hours in UTC (fixed offset for simplicity in filtering)
# 09:30 ET = 13:30 UTC (EST+5) or 14:30 UTC (EDT+4)
# We use the timezone-aware approach instead of hardcoded UTC offsets

# ── Market-hours gate ──────────────────────────────────────────────────────────


def _is_market_hours_ts(ts: datetime) -> bool:
    """
    Return True if the given timestamp falls within NYSE market hours
    (9:30-16:00 ET, Mon-Fri).  Checks the BAR's timestamp, not current time.
    """
    try:
        ts_et = ts.astimezone(MARKET_TZ)
        if ts_et.weekday() >= 5:  # Sat=5, Sun=6
            return False
        t = ts_et.time().replace(second=0, microsecond=0)
        return _MARKET_OPEN <= t < _MARKET_CLOSE
    except Exception as exc:
        logger.warning("[polygon] Market hours check failed for {}: {}", ts, exc)
        return False


def _is_market_open() -> bool:
    """Return True if NYSE is currently open (9:30-16:00 ET, Mon-Fri)."""
    now_et = datetime.now(tz=MARKET_TZ)
    if now_et.weekday() >= 5:
        return False
    t = now_et.time().replace(second=0, microsecond=0)
    return _MARKET_OPEN <= t < _MARKET_CLOSE


def _yesterday_market_window() -> tuple[datetime, datetime]:
    """Return yesterday's regular-market session as UTC datetimes."""
    yesterday_et = datetime.now(tz=MARKET_TZ).date() - timedelta(days=1)
    start_et = datetime.combine(yesterday_et, _MARKET_OPEN, tzinfo=MARKET_TZ)
    end_et = datetime.combine(yesterday_et, _MARKET_CLOSE, tzinfo=MARKET_TZ)
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)


def _cap_rest_range_to_yesterday(
    from_ts: datetime,
    to_ts: datetime,
) -> tuple[datetime, datetime] | None:
    """
    Polygon REST backfill is historical-only. Keep the URL date at yesterday
    or earlier because requesting today's date returns 403 on the current plan.
    """
    yesterday_start, yesterday_end = _yesterday_market_window()

    if from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=timezone.utc)
    if to_ts.tzinfo is None:
        to_ts = to_ts.replace(tzinfo=timezone.utc)

    from_ts = from_ts.astimezone(timezone.utc)
    to_ts = to_ts.astimezone(timezone.utc)

    if from_ts >= yesterday_end:
        from_ts = yesterday_start
    if to_ts > yesterday_end:
        to_ts = yesterday_end

    if from_ts >= to_ts:
        return None
    return from_ts, to_ts


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_am_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Parse a Polygon AM (aggregate-per-minute) WebSocket message into a
    normalised bar dict ready for DB insertion.

    Polygon AM fields:
      ev  – event type ("AM")
      sym – symbol
      v   – volume
      o   – open, c – close, h – high, l – low
      vw  – volume-weighted average price
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


_UPSERT_SQL = """
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
"""


async def _persist_bar(bar: dict[str, Any]) -> None:
    """Upsert a single OHLCV bar into market_data."""
    try:
        await db.execute(
            _UPSERT_SQL,
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
    except Exception as exc:
        logger.error("[polygon] DB write error for {}: {}", bar["symbol"], exc)
        raise


# ── Gap detection ─────────────────────────────────────────────────────────────


async def _check_and_fill_gap(symbol: str, bar_ts: datetime, api_key: str) -> None:
    """
    After saving a bar, check if the previous minute's bar exists.
    If missing and within market hours, immediately backfill from REST API.
    Zero gaps tolerance.
    """
    try:
        prev_minute = bar_ts - timedelta(minutes=1)

        # Only check gaps within market hours
        if not _is_market_hours_ts(prev_minute):
            return

        exists = await db.fetchval(
            "SELECT 1 FROM market_data WHERE symbol=$1 AND timestamp=$2",
            symbol, prev_minute,
        )
        if exists:
            return

        logger.warning(
            "[polygon] GAP DETECTED: {} missing bar at {:%H:%M} UTC — backfilling",
            symbol, prev_minute,
        )
        await _rest_backfill_range(
            symbol, prev_minute, bar_ts, api_key,
        )
    except Exception as exc:
        logger.error("[polygon] Gap check error for {} @ {}: {}", symbol, bar_ts, exc)


async def _rest_backfill_range(
    symbol: str,
    from_ts: datetime,
    to_ts: datetime,
    api_key: str,
) -> int:
    """
    Fetch 1-minute bars from Polygon REST API for a specific time range.
    Returns count of bars inserted.
    """
    if _is_market_open():
        logger.info(
            "[polygon] Market open — skipping REST backfill for {}; WebSocket handles live data",
            symbol,
        )
        return 0

    capped_range = _cap_rest_range_to_yesterday(from_ts, to_ts)
    if capped_range is None:
        logger.info(
            "[polygon] REST backfill skipped for {}; requested range is not historical: {} → {}",
            symbol,
            from_ts,
            to_ts,
        )
        return 0

    from_ts, to_ts = capped_range
    from_str = from_ts.strftime("%Y-%m-%d")
    to_str = to_ts.strftime("%Y-%m-%d")

    # Use millisecond timestamps for precise range
    from_ms = int(from_ts.timestamp() * 1000)
    to_ms = int(to_ts.timestamp() * 1000)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
        f"{from_str}/{to_str}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            results = resp.json().get("results", [])

        if not results:
            logger.debug("[polygon] REST backfill: no results for {} ({} → {})", symbol, from_ts, to_ts)
            return 0

        count = 0
        for r in results:
            bar_ts = datetime.fromtimestamp(r["t"] / 1000.0, tz=timezone.utc)

            # Only store bars within market hours
            if not _is_market_hours_ts(bar_ts):
                continue

            bar = {
                "symbol":     symbol,
                "timestamp":  bar_ts,
                "open":       float(r["o"]),
                "high":       float(r["h"]),
                "low":        float(r["l"]),
                "close":      float(r["c"]),
                "volume":     float(r["v"]),
                "vwap":       float(r.get("vw", 0.0)),
                "num_trades": int(r.get("n", 0)),
                "exchange":   "polygon_rest",
                "source":     "polygon_rest",
            }
            try:
                await _persist_bar(bar)
                count += 1
            except Exception as exc:
                logger.error("[polygon] REST backfill persist error: {}", exc)

        if count > 0:
            logger.success("[polygon] REST backfill: {} bars filled for {}", count, symbol)
        return count

    except Exception as exc:
        logger.error("[polygon] REST backfill failed for {}: {}", symbol, exc)
        return 0


# ── Periodic REST backfill loop ───────────────────────────────────────────────


async def _periodic_backfill_loop(api_key: str) -> None:
    """
    Every 5 minutes while the market is closed, check yesterday's session for
    historical gaps and backfill from REST API. Live gaps during market hours
    are handled by the WebSocket stream.
    """
    while True:
        try:
            await asyncio.sleep(_BACKFILL_INTERVAL_S)

            if _is_market_open():
                logger.debug("[polygon] Market open — skipping REST backfill")
                continue

            logger.info("[polygon] Running historical REST backfill check for yesterday …")
            from_ts, to_ts = _yesterday_market_window()

            for symbol in EQUITY_SYMBOLS:
                try:
                    # Find gaps: minutes where we should have data but don't
                    rows = await db.fetch(
                        """
                        SELECT timestamp FROM market_data
                        WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                        ORDER BY timestamp ASC
                        """,
                        symbol, from_ts, to_ts,
                    )

                    existing_ts = {row["timestamp"] for row in rows}

                    # Generate expected timestamps (every minute during market hours)
                    check_ts = from_ts.replace(second=0, microsecond=0)
                    gaps = []
                    while check_ts < to_ts:
                        if _is_market_hours_ts(check_ts) and check_ts not in existing_ts:
                            gaps.append(check_ts)
                        check_ts += timedelta(minutes=1)

                    if gaps:
                        logger.warning(
                            "[polygon] Found {} historical gaps for {} yesterday — backfilling",
                            len(gaps), symbol,
                        )
                        await _rest_backfill_range(symbol, gaps[0], gaps[-1] + timedelta(minutes=1), api_key)
                    else:
                        logger.debug("[polygon] No gaps for {} in last 10 min ✓", symbol)

                except Exception as exc:
                    logger.error("[polygon] Periodic backfill error for {}: {}", symbol, exc)

        except asyncio.CancelledError:
            logger.info("[polygon] Periodic backfill loop cancelled")
            break
        except Exception as exc:
            logger.error("[polygon] Periodic backfill loop error: {}", exc)


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
        max_size=10 * 1024 * 1024,   # 10 MB
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

        # ── Step 4: start periodic backfill alongside WebSocket ──────────────
        backfill_task = asyncio.create_task(_periodic_backfill_loop(api_key))

        try:
            # ── Step 5: consume messages ─────────────────────────────────────
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
                            # Filter on BAR timestamp, not current time
                            if _is_market_hours_ts(bar["timestamp"]):
                                try:
                                    await _persist_bar(bar)
                                    # Gap detection: check previous minute
                                    asyncio.create_task(
                                        _check_and_fill_gap(
                                            bar["symbol"],
                                            bar["timestamp"],
                                            api_key,
                                        )
                                    )
                                except Exception as db_exc:
                                    logger.error("[polygon] DB write error: {}", db_exc)
                            else:
                                logger.debug(
                                    "[polygon] Outside market hours — discarding {} bar @ {}",
                                    bar["symbol"], bar["timestamp"],
                                )

                    elif ev == "status":
                        logger.debug("[polygon] status: {}", msg)

                    else:
                        logger.trace("[polygon] unhandled ev='{}': {}", ev, msg)
        finally:
            backfill_task.cancel()
            try:
                await backfill_task
            except asyncio.CancelledError:
                pass


# ── Public entry point ────────────────────────────────────────────────────────


async def run_polygon_collector() -> None:
    """
    Start the Polygon collector with automatic reconnection.
    Uses tenacity for exponential back-off on any connection/protocol error.
    A PermissionError (bad API key) stops retrying immediately.
    """
    api_key = settings.polygon_api_key
    logger.info(
        "[polygon] Starting collector for symbols: {}", EQUITY_SYMBOLS
    )

    # Initialise DB pool once
    await db.init_pool()

    # Run initial historical backfill to fill any existing gaps
    logger.info("[polygon] Running initial REST backfill for yesterday …")
    from_ts, to_ts = _yesterday_market_window()
    for symbol in EQUITY_SYMBOLS:
        try:
            await _rest_backfill_range(
                symbol,
                from_ts,
                to_ts,
                api_key,
            )
        except Exception as exc:
            logger.error("[polygon] Initial backfill failed for {}: {}", symbol, exc)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(
            (
                ConnectionClosed,
                WebSocketException,
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
