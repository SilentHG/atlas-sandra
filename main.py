"""
ATLAS System Entry Point
=========================
Wires all agents together and starts the event loop.

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from agents.orchestrator import OrchestratorAgent
from data_ingestion.ingestor_agent import DataIngestorAgent
from database.connection import close_pool, init_pool
from config.settings import settings

# ── Logging setup ─────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
    colorize=True,
)
logger.add(
    "logs/atlas_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="14 days",
    level="DEBUG",
    enqueue=True,
)

# ── Watchlist ─────────────────────────────────────────────────
WATCHLIST = ["AAPL", "TSLA", "NVDA", "SPY", "BTC/USDT"]


async def main() -> None:
    logger.info("═══════════════════════════════════════")
    logger.info("  ATLAS Trading System — Day 1 Boot    ")
    logger.info("  Environment: {}", settings.environment)
    logger.info("═══════════════════════════════════════")

    # 1. Database pool
    await init_pool()

    # 2. Build agents
    orchestrator  = OrchestratorAgent()
    data_ingestor = DataIngestorAgent(symbols=WATCHLIST, config={"tick_seconds": 60})

    orchestrator.register_agent(data_ingestor)

    # 3. Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("Received {}, shutting down …", sig_name)
        loop.create_task(_graceful_shutdown(orchestrator))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig.name)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    # 4. Start
    await orchestrator.start()


async def _graceful_shutdown(orchestrator: OrchestratorAgent) -> None:
    await orchestrator.stop()
    await close_pool()
    logger.info("ATLAS shutdown complete. Goodbye.")
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    import pathlib
    pathlib.Path("logs").mkdir(exist_ok=True)
    asyncio.run(main())
