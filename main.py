"""
ATLAS System Entry Point  ─ Day 2
===================================
Wires all agents together and starts the async event loop.

Day 1  agents: DataIngestorAgent (Polygon + Binance WebSocket)
Day 2  agents: FeatureEngineerAgent (50+ indicators every 60 s)

Strategy generation (ideator + coder) runs once on startup if no active
strategies exist, then the system enters normal operation.

Usage:
    python main.py
    python main.py --skip-ideate   # skip Claude calls (use existing strategies)
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from loguru import logger

from config.settings import settings
from database.connection import close_pool, init_pool

# ── Logging ──────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan> — <level>{message}</level>"
    ),
    colorize=True,
)
logger.add(
    "logs/atlas_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="14 days",
    level="DEBUG",
    enqueue=True,
)

# ── Watchlist ─────────────────────────────────────────────────────────────────

EQUITY_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
CRYPTO_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FULL_WATCHLIST   = EQUITY_WATCHLIST + CRYPTO_WATCHLIST


# ── Bootstrap helpers ─────────────────────────────────────────────────────────


async def _count_active_strategies() -> int:
    """Return number of rows in strategies with status='active'."""
    try:
        from database import connection as db
        rows = await db.fetch("SELECT COUNT(*) AS n FROM strategies WHERE status = 'active'")
        return int(rows[0]["n"]) if rows else 0
    except Exception:
        return 0


async def _bootstrap_strategies() -> None:
    """
    If no active strategies exist, run the ideator then the coder to
    generate and activate 2 Claude-written strategies.
    """
    n = await _count_active_strategies()
    if n > 0:
        logger.info("[main] {} active strategy/strategies found — skipping ideation.", n)
        return

    logger.info("[main] No active strategies found — running ideator …")
    try:
        from strategy_engine.ideator import run_ideator
        ids = await run_ideator(n=10)
        if ids:
            logger.info("[main] Ideator created {} draft strategies.", len(ids))
        else:
            logger.warning("[main] Ideator returned no strategies — check API key / logs.")
            return
    except Exception as exc:
        logger.error("[main] Ideator failed: {}", exc, exc_info=True)
        return

    logger.info("[main] Running strategy coder …")
    try:
        from strategy_engine.strategy_coder import run_strategy_coder
        coded = await run_strategy_coder()
        logger.info("[main] Strategy coder activated {} strategies.", len(coded))
    except Exception as exc:
        logger.error("[main] Strategy coder failed: {}", exc, exc_info=True)


# ── Main ───────────────────────────────────────────────────────────────────────


async def main(skip_ideate: bool = False) -> None:
    logger.info("═══════════════════════════════════════════════")
    logger.info("  ATLAS Trading System — Day 2 Boot           ")
    logger.info("  Environment : {}", settings.environment)
    logger.info("  Symbols     : {}", FULL_WATCHLIST)
    logger.info("═══════════════════════════════════════════════")

    # 1. Database pool
    await init_pool()
    logger.info("[main] Database pool initialised.")

    # 1b. Kill switch setup (restore state from DB)
    from risk_management.kill_switch import get_kill_switch
    ks = get_kill_switch()
    await ks.setup()

    # 2. Strategy bootstrap (ideate + code if no active strategies)
    if not skip_ideate:
        await _bootstrap_strategies()
    else:
        logger.info("[main] --skip-ideate flag set — skipping Claude strategy generation.")

    # 3. Build agents
    from agents.orchestrator import OrchestratorAgent
    from data_ingestion.ingestor_agent import DataIngestorAgent
    from feature_store.feature_engineer import FeatureEngineerAgent

    orchestrator    = OrchestratorAgent()
    data_ingestor   = DataIngestorAgent(symbols=FULL_WATCHLIST, config={"tick_seconds": 60})
    feature_engineer = FeatureEngineerAgent(config={"tick_seconds": 60})

    orchestrator.register_agent(data_ingestor)
    orchestrator.register_agent(feature_engineer)

    logger.info("[main] Agents registered: data_ingestor, feature_engineer")

    # 4. Graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(sig_name: str) -> None:
        logger.info("[main] {} received — initiating graceful shutdown …", sig_name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig.name)
        except NotImplementedError:
            pass   # Windows: handled via KeyboardInterrupt

    # 5. Start
    orchestrator_task = asyncio.create_task(orchestrator.start())

    try:
        await asyncio.wait(
            [orchestrator_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
    except KeyboardInterrupt:
        logger.info("[main] KeyboardInterrupt — shutting down …")
    finally:
        await _graceful_shutdown(orchestrator)
        orchestrator_task.cancel()
        try:
            await orchestrator_task
        except asyncio.CancelledError:
            pass


async def _graceful_shutdown(orchestrator: Any) -> None:
    try:
        await orchestrator.stop()
    except Exception:
        pass
    await close_pool()
    logger.info("[main] ATLAS shutdown complete. Goodbye.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import Any
    parser = argparse.ArgumentParser(description="ATLAS Trading System")
    parser.add_argument(
        "--skip-ideate", action="store_true",
        help="Skip Claude strategy ideation/coding on startup",
    )
    args = parser.parse_args()
    asyncio.run(main(skip_ideate=args.skip_ideate))
