"""
ATLAS Reddit Scout — scouts/reddit_scout.py
============================================
Scans r/algotrading and r/stocks for strategy ideas
and extracts hypotheses using Claude.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from anthropic import Anthropic
from loguru import logger

from config.settings import settings
from database import connection as db

SUBREDDITS = ["algotrading", "stocks", "investing", "quant"]

_EXTRACT_PROMPT = """You are a systematic trading researcher.
Given this Reddit post, extract a trading strategy hypothesis if one exists.

Title: {title}
Body: {body}

If no clear strategy exists, return {{"confidence": 0.0}}.
Otherwise return JSON ONLY:
{{
  "description": "brief strategy description",
  "entry_rules": ["rule 1"],
  "exit_rules":  ["rule 1"],
  "confidence":  0.0-1.0
}}"""


class RedditScout:
    """Scans Reddit for strategy ideas and extracts them via Claude."""

    _HEADERS = {"User-Agent": "ATLAS-Scout/1.0 (trading research bot)"}

    def __init__(self) -> None:
        self._claude = Anthropic(api_key=settings.anthropic_api_key)

    async def setup(self) -> None:
        # Reuse same table created by YouTubeScout
        await db.execute("""
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
        """)

    async def scan(
        self,
        subreddits: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        targets = subreddits or SUBREDDITS
        hypotheses = []
        async with httpx.AsyncClient(headers=self._HEADERS, timeout=20.0) as client:
            for sub in targets:
                posts = await self._fetch_posts(client, sub, limit)
                for post in posts:
                    h = await self._extract(post)
                    if h and h.get("confidence", 0) >= 0.3:
                        await self._save(h)
                        hypotheses.append(h)
        return hypotheses

    async def _fetch_posts(self, client: httpx.AsyncClient, sub: str, limit: int) -> list[dict]:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
            r = await client.get(url)
            r.raise_for_status()
            children = r.json().get("data", {}).get("children", [])
            return [
                {
                    "title":    c["data"].get("title", ""),
                    "body":     (c["data"].get("selftext", "") or "")[:1000],
                    "url":      f"https://reddit.com{c['data'].get('permalink','')}",
                    "subreddit": sub,
                    "score":    c["data"].get("score", 0),
                }
                for c in children
                if not c["data"].get("stickied")
            ]
        except Exception as exc:
            logger.warning("[reddit_scout] Failed to fetch r/{}: {}", sub, exc)
            return []

    async def _extract(self, post: dict) -> dict | None:
        if len(post.get("body", "")) < 50 and len(post.get("title", "")) < 30:
            return None
        try:
            msg = self._claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": _EXTRACT_PROMPT.format(
                        title=post["title"], body=post.get("body", ""),
                    ),
                }],
            )
            raw  = msg.content[0].text.strip()
            data = json.loads(raw)
            if data.get("confidence", 0) < 0.3:
                return None
            return {
                "source":      "reddit",
                "source_url":  post.get("url"),
                "description": data.get("description"),
                "entry_rules": data.get("entry_rules", []),
                "exit_rules":  data.get("exit_rules", []),
                "confidence":  float(data.get("confidence", 0.3)),
                "scout":       "reddit_scout",
                "raw_metadata": post,
            }
        except Exception as exc:
            logger.debug("[reddit_scout] Extract failed: {}", exc)
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
            logger.info("[reddit_scout] Saved: {:.60}", h["description"] or "")
        except Exception as exc:
            logger.error("[reddit_scout] Save: {}", exc)
