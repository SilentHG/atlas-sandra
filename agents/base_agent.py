"""
ATLAS Base Agent
=================
Abstract base class for all ATLAS agents.
Every agent inherits lifecycle hooks: setup, run, teardown, heartbeat.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from database import connection as db


class BaseAgent(ABC):
    """Abstract base class for all ATLAS agents."""

    agent_type: str = "base"

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.id: uuid.UUID = uuid.uuid4()
        self.name: str = name
        self.config: dict[str, Any] = config or {}
        self._running: bool = False
        self._heartbeat_interval: int = self.config.get("heartbeat_interval_s", 30)

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        """Start the agent: setup → run loop → heartbeat."""
        logger.info("[{}] Starting …", self.name)
        await self._register()
        await self.setup()
        self._running = True
        await self._update_status("running")
        await asyncio.gather(
            self._run_loop(),
            self._heartbeat_loop(),
        )

    async def stop(self) -> None:
        """Gracefully stop the agent."""
        logger.info("[{}] Stopping …", self.name)
        self._running = False
        await self.teardown()
        await self._update_status("stopped")

    # ── Abstract hooks ────────────────────────────────────────

    async def setup(self) -> None:
        """Override for one-time initialisation (subscriptions, warm-up, etc.)."""

    @abstractmethod
    async def run(self) -> None:
        """Core agent logic — called once per loop iteration."""

    async def teardown(self) -> None:
        """Override for cleanup (close connections, flush queues, etc.)."""

    # ── Internal loops ────────────────────────────────────────

    async def _run_loop(self) -> None:
        tick_s = self.config.get("tick_seconds", 1)
        while self._running:
            try:
                await self.run()
            except Exception as exc:
                logger.exception("[{}] Error in run(): {}", self.name, exc)
                await self._increment_error(str(exc))
            await asyncio.sleep(tick_s)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            await self._send_heartbeat()

    # ── Database helpers ──────────────────────────────────────

    async def _register(self) -> None:
        """Upsert this agent into the agent_registry table."""
        try:
            await db.execute(
                """
                INSERT INTO agent_registry
                    (name, agent_type, config, status, last_heartbeat)
                VALUES ($1, $2, $3::jsonb, 'idle', NOW())
                ON CONFLICT (name) DO UPDATE
                    SET config         = EXCLUDED.config,
                        status         = 'idle',
                        updated_at     = NOW()
                """,
                self.name,
                self.agent_type,
                str(self.config).replace("'", '"'),
            )
        except Exception as exc:
            logger.warning("[{}] Could not register in DB: {}", self.name, exc)

    async def _update_status(self, status: str) -> None:
        try:
            await db.execute(
                "UPDATE agent_registry SET status=$1, updated_at=NOW() WHERE name=$2",
                status,
                self.name,
            )
        except Exception as exc:
            logger.warning("[{}] Could not update status: {}", self.name, exc)

    async def _send_heartbeat(self) -> None:
        try:
            await db.execute(
                "UPDATE agent_registry SET last_heartbeat=NOW() WHERE name=$1",
                self.name,
            )
        except Exception as exc:
            logger.warning("[{}] Heartbeat failed: {}", self.name, exc)

    async def _increment_error(self, message: str) -> None:
        try:
            await db.execute(
                """
                UPDATE agent_registry
                   SET error_count = error_count + 1,
                       last_error  = $1,
                       status      = 'error',
                       updated_at  = NOW()
                 WHERE name        = $2
                """,
                message[:500],
                self.name,
            )
        except Exception:
            pass

    async def log(self, level: str, message: str, metadata: dict | None = None) -> None:
        """Persist a structured log entry to the agent_logs table."""
        logger.log(level.upper(), "[{}] {}", self.name, message)
        try:
            await db.execute(
                """
                INSERT INTO agent_logs (agent_name, level, message, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                self.name,
                level.upper(),
                message,
                str(metadata or {}).replace("'", '"'),
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} running={self._running}>"
