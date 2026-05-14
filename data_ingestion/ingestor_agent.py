"""
ATLAS Data Ingestor Agent
==========================
Orchestrates the data ingestion layer. Delegates to:
  - data_ingestion.polygon_collector  (Polygon.io WebSocket)
  - data_ingestion.binance_collector  (Binance WebSocket)
  - data_ingestion.alpaca_connector   (Alpaca order management)

This stub runs a lightweight REST-based poll for backwards compatibility.
For production use, launch the WebSocket collectors directly.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from agents.base_agent import BaseAgent
from config.settings import settings
from database import connection as db


class DataIngestorAgent(BaseAgent):
    """Ingests live market data from Polygon and Binance."""

    agent_type = "data"

    def __init__(self, symbols: list[str], config: dict[str, Any] | None = None) -> None:
        super().__init__(name="data_ingestor", config=config)
        self.symbols = symbols
        self._session: aiohttp.ClientSession | None = None

    async def setup(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info("[data_ingestor] Ready for symbols: {}", self.symbols)

    async def run(self) -> None:
        """Fetch the latest bar for each symbol via Polygon REST and persist it."""
        for symbol in self.symbols:
            try:
                bar = await self._fetch_polygon_bar(symbol)
                if bar:
                    await self._persist_bar(symbol, bar, source="polygon")
            except Exception as exc:
                logger.warning("[data_ingestor] Failed to fetch {}: {}", symbol, exc)

    async def teardown(self) -> None:
        if self._session:
            await self._session.close()

    # ── Polygon helpers ───────────────────────────────────────

    async def _fetch_polygon_bar(self, symbol: str) -> dict | None:
        """Fetch the latest 1-minute bar from Polygon REST API."""
        api_key = settings.polygon_api_key
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            f"?adjusted=true&apiKey={api_key}"
        )
        if not self._session:
            return None
        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    return results[0] if results else None
        except Exception as exc:
            logger.debug("[data_ingestor] Polygon request error: {}", exc)
        return None

    async def _persist_bar(self, symbol: str, bar: dict, source: str) -> None:
        """Insert an OHLCV bar into market_data."""
        ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
        await db.execute(
            """
            INSERT INTO market_data
                (symbol, timestamp, open, high, low, close,
                 volume, vwap, num_trades, exchange, source)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (symbol, timestamp) DO NOTHING
            """,
            symbol,
            ts,
            bar.get("o"),
            bar.get("h"),
            bar.get("l"),
            bar.get("c"),
            bar.get("v", 0.0),
            bar.get("vw"),
            bar.get("n"),
            bar.get("x", "polygon"),
            source,
        )
        logger.debug("[data_ingestor] Saved bar {}: close={}", symbol, bar.get("c"))
