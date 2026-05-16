"""
ATLAS Orchestrator — agents/orchestrator.py
============================================
Spawns, pauses, resumes, and kills agents.
Sends heartbeats every 30 s. Cleans up dead agents from registry.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent
from database import connection as db


class OrchestratorAgent(BaseAgent):
    """Manages all child agents and enforces system health."""

    agent_type = "orchestrator"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(
            name="orchestrator",
            config={"tick_seconds": 30, "heartbeat_interval_s": 30, **(config or {})},
        )
        self._agents:  dict[str, BaseAgent]  = {}
        self._tasks:   dict[str, asyncio.Task] = {}
        self._paused:  set[str]              = set()

    # ── Registration ──────────────────────────────────────────────────────────

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        logger.info("[orchestrator] Registered: {}", agent.name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        await db.init_pool()
        logger.info("[orchestrator] Starting {} child agent(s)…", len(self._agents))
        for agent in self._agents.values():
            self._tasks[agent.name] = asyncio.create_task(agent.start())

    async def run(self) -> None:
        """Health-check + dead-agent cleanup every tick."""
        for name, agent in list(self._agents.items()):
            task = self._tasks.get(name)
            if name in self._paused:
                continue
            if task is None or task.done():
                if task and task.exception():
                    logger.error("[orchestrator] {} crashed: {}", name, task.exception())
                logger.warning("[orchestrator] Restarting {}", name)
                self._tasks[name] = asyncio.create_task(agent.start())

        await self._cleanup_dead_registry()

    async def teardown(self) -> None:
        logger.info("[orchestrator] Stopping all agents…")
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await asyncio.gather(*[a.stop() for a in self._agents.values()], return_exceptions=True)

    # ── Dynamic spawn / pause / resume / kill ─────────────────────────────────

    async def spawn(self, agent: BaseAgent) -> str:
        """Dynamically spawn a new agent and register it."""
        self.register_agent(agent)
        self._tasks[agent.name] = asyncio.create_task(agent.start())
        logger.info("[orchestrator] Spawned: {}", agent.name)
        return agent.name

    async def pause(self, agent_name: str) -> bool:
        agent = self._agents.get(agent_name)
        if not agent:
            return False
        agent._running = False
        self._paused.add(agent_name)
        await db.execute(
            "UPDATE agent_registry SET status='paused', updated_at=NOW() WHERE name=$1",
            agent_name,
        )
        logger.info("[orchestrator] Paused: {}", agent_name)
        return True

    async def resume(self, agent_name: str) -> bool:
        agent = self._agents.get(agent_name)
        if not agent or agent_name not in self._paused:
            return False
        self._paused.discard(agent_name)
        self._tasks[agent_name] = asyncio.create_task(agent.start())
        logger.info("[orchestrator] Resumed: {}", agent_name)
        return True

    async def kill(self, agent_name: str) -> bool:
        agent = self._agents.get(agent_name)
        if not agent:
            return False
        task = self._tasks.pop(agent_name, None)
        if task:
            task.cancel()
        await agent.stop()
        del self._agents[agent_name]
        self._paused.discard(agent_name)
        logger.warning("[orchestrator] Killed: {}", agent_name)
        return True

    # ── Registry helpers ──────────────────────────────────────────────────────

    async def get_registry(self) -> list[dict]:
        rows = await db.fetch(
            "SELECT name,agent_type,status,last_heartbeat,error_count,last_error FROM agent_registry ORDER BY name"
        )
        return [dict(r) for r in rows]

    async def get_agent_status(self, agent_name: str) -> dict | None:
        row = await db.fetchrow(
            "SELECT * FROM agent_registry WHERE name=$1", agent_name
        )
        return dict(row) if row else None

    async def _cleanup_dead_registry(self) -> None:
        """Remove registry entries for agents not in memory."""
        try:
            known = list(self._agents.keys()) + ["orchestrator"]
            await db.execute(
                "UPDATE agent_registry SET status='stopped', updated_at=NOW() "
                "WHERE name != ALL($1::text[]) AND status='running' "
                "AND last_heartbeat < NOW() - INTERVAL '2 minutes'",
                known,
            )
        except Exception as exc:
            logger.warning("[orchestrator] Registry cleanup error: {}", exc)
