"""
ATLAS YouTube Scout — scouts/youtube_scout.py
==============================================
Searches YouTube for trading strategy videos and
extracts hypotheses using Claude.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from anthropic import Anthropic
from loguru import logger

from config.settings import settings
from database import connection as db

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS strategy_hypotheses (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source       TEXT NOT NULL,
    source_url   TEXT,
    description  TEXT,
    entry_rules  JSONB,
    exit_rules   JSONB,
    confidence   DOUBLE PRECISION DEFAULT 0.0,
    scout        TEXT NOT NULL DEFAULT 'unknown',
    raw_metadata JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_EXTRACT_PROMPT = """You are a systematic trading researcher.
Given the following video title and description, extract a trading strategy hypothesis.

Source: {title}
Description: {description}

Return a JSON object with these fields ONLY:
{{
  "description": "brief strategy description",
  "entry_rules": ["rule 1", "rule 2"],
  "exit_rules": ["rule 1", "rule 2"],
  "confidence": 0.0-1.0
}}
Return ONLY valid JSON, no markdown."""


class YouTubeScout:
    """Searches YouTube and extracts strategy hypotheses via Claude."""

    def __init__(self) -> None:
        self._yt_key  = getattr(settings, "youtube_api_key", None)
        self._claude  = Anthropic(api_key=settings.anthropic_api_key)

    async def setup(self) -> None:
        await db.execute(_ENSURE_TABLE)

    async def search_and_extract(
        self,
        query: str = "algorithmic trading strategy python",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        videos = await self._search_youtube(query, max_results)
        hypotheses = []
        for v in videos:
            h = await self._extract(v)
            if h:
                await self._save(h)
                hypotheses.append(h)
        return hypotheses

    async def _search_youtube(self, query: str, max_results: int) -> list[dict]:
        if not self._yt_key:
            raise RuntimeError("YOUTUBE_API_KEY is required for YouTube scout extraction.")
        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={query}&type=video&maxResults={max_results}"
            f"&relevanceLanguage=en&key={self._yt_key}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        return [
            {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "url": f"https://youtube.com/watch?v={item['id']['videoId']}",
            }
            for item in data.get("items", [])
        ]

    async def _extract(self, video: dict) -> dict | None:
        try:
            msg = self._claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": _EXTRACT_PROMPT.format(
                        title=video["title"],
                        description=video.get("description", ""),
                    ),
                }],
            )
            raw = msg.content[0].text.strip()
            data = json.loads(raw)
            return {
                "source": "youtube",
                "source_url": video.get("url"),
                "description": data.get("description"),
                "entry_rules": data.get("entry_rules", []),
                "exit_rules":  data.get("exit_rules", []),
                "confidence":  float(data.get("confidence", 0.5)),
                "scout":       "youtube_scout",
                "raw_metadata": video,
            }
        except Exception as exc:
            logger.error("[youtube_scout] Extract failed for '{}': {}", video.get("title"), exc)
            return None

    async def _save(self, h: dict) -> None:
        try:
            await db.execute(
                """INSERT INTO strategy_hypotheses
                   (source,source_url,description,entry_rules,exit_rules,confidence,scout,raw_metadata)
                   VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7,$8::jsonb)""",
                h["source"], h["source_url"], h["description"],
                json.dumps(h["entry_rules"]), json.dumps(h["exit_rules"]),
                h["confidence"], h["scout"], json.dumps(h["raw_metadata"]),
            )
            logger.info("[youtube_scout] Saved: {:.60}", h["description"])
        except Exception as exc:
            logger.error("[youtube_scout] Save failed: {}", exc)
