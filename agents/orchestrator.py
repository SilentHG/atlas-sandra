"""
ATLAS Orchestrator Agent
=========================
Top-level coordinator: manages agent lifecycle, health checks,
and restart policy for the whole system.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """Manages all child agents and enforces system health."""

    agent_type = "orchestrator"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="orchestrator", config=config)
        self._agents: dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a child agent under the orchestrator."""
        self._agents[agent.name] = agent
        logger.info("[orchestrator] Registered agent: {}", agent.name)

    async def setup(self) -> None:
        logger.info("[orchestrator] Starting {} child agent(s) …", len(self._agents))
        tasks = [asyncio.create_task(a.start()) for a in self._agents.values()]
        # Store tasks so they are not garbage-collected
        self._tasks = tasks

    async def run(self) -> None:
        """Health-check: restart any crashed agents."""
        for name, agent in self._agents.items():
            if not agent._running:
                logger.warning("[orchestrator] Agent '{}' is down — restarting …", name)
                asyncio.create_task(agent.start())

    async def teardown(self) -> None:
        logger.info("[orchestrator] Stopping all child agents …")
        await asyncio.gather(*[a.stop() for a in self._agents.values()])
