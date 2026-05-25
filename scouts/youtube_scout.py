"""
ATLAS YouTube Scout — scouts/youtube_scout.py
==============================================
Searches YouTube for trading strategy videos and
extracts hypotheses using Claude.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, parse_qs

import httpx
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
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

_SUMMARY_PROMPT = """You are a trading research assistant.
Summarize the following YouTube transcript into the core trading idea only.

Rules:
- Focus on the strategy logic from the video content.
- Ignore promotions, disclaimers, intros, links, and filler.
- Extract market, setup, indicators, entry logic, exit logic, risk logic.
- Keep it concise.

Title: {title}

Transcript:
{transcript}
"""

_EXTRACT_PROMPT = """You are a systematic trading researcher.

Primary source:
The strategy idea MUST come from the YouTube transcript summary below.

Reference only:
ATLAS feature names, indicators, and feature-store context may be used only as supporting references. Do not invent a strategy from features alone.

Video title: {title}
Transcript summary:
{summary}

Reference features available:
{features_reference}

Return a JSON object with these fields ONLY:
{{
  "description": "brief strategy description based on the video transcript",
  "entry_rules": ["rule 1", "rule 2"],
  "exit_rules": ["rule 1", "rule 2"],
  "confidence": 0.0-1.0
}}
Return ONLY valid JSON, no markdown."""


def _video_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.netloc in {"youtube.com", "www.youtube.com"}:
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.netloc == "youtu.be":
            return parsed.path.strip("/") or None
    except Exception:
        return None
    return None


def _clean_transcript_text(text: str, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


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
            return [
                {
                    "title": "Momentum Trading Strategy With VWAP and RSI",
                    "description": "Buy when price reclaims VWAP, RSI crosses above 50, and volume expands. Exit on VWAP loss or ATR stop.",
                    "url": "youtube://fallback/momentum-vwap-rsi",
                    "channel": "ATLAS fallback scout",
                },
                {
                    "title": "Breakout Strategy Using Volume Confirmation",
                    "description": "Enter breakouts above prior high only when volume is above 2x average and trend filter is positive.",
                    "url": "youtube://fallback/breakout-volume",
                    "channel": "ATLAS fallback scout",
                },
                {
                    "title": "Mean Reversion Bollinger Band Strategy",
                    "description": "Enter oversold Bollinger Band moves when RSI confirms exhaustion. Exit at mid-band or stop below low.",
                    "url": "youtube://fallback/bb-rsi-mean-reversion",
                    "channel": "ATLAS fallback scout",
                },
            ][:max_results]
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
                "video_id": item["id"]["videoId"],
            }
            for item in data.get("items", [])
        ]

    async def _already_seen(self, source_url: str | None) -> bool:
        if not source_url:
            return False
        row = await db.fetchrow(
            "SELECT id FROM strategy_hypotheses WHERE source = 'youtube' AND source_url = $1 LIMIT 1",
            source_url,
        )
        return row is not None

    async def _get_transcript(self, video: dict) -> str:
        video_id = video.get("video_id") or _video_id_from_url(video.get("url"))
        if not video_id:
            return ""

        def _load() -> str:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en"])
            return " ".join(snippet.text for snippet in fetched)

        import asyncio
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, _load)
            return _clean_transcript_text(raw)
        except Exception as exc:
            logger.warning("[youtube_scout] Transcript unavailable for '{}': {}", video.get("title"), exc)
            return ""

    async def _summarize_transcript(self, title: str, transcript: str) -> str:
        if not transcript:
            return ""

        def _call() -> str:
            msg = self._claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=900,
                messages=[{
                    "role": "user",
                    "content": _SUMMARY_PROMPT.format(title=title, transcript=transcript),
                }],
            )
            return msg.content[0].text.strip()

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call)

    async def _feature_reference(self) -> str:
        try:
            rows = await db.fetch(
                """
                SELECT DISTINCT feature_name
                FROM feature_store
                WHERE computed_at > NOW() - INTERVAL '30 minutes'
                ORDER BY feature_name
                LIMIT 80
                """
            )
            names = [r["feature_name"] for r in rows]
            return ", ".join(names[:80])
        except Exception as exc:
            logger.warning("[youtube_scout] Feature reference unavailable: {}", exc)
            return "EMA, RSI, MACD, VWAP, ATR, Bollinger Bands, volume, regime, volatility"

    async def _extract(self, video: dict) -> dict | None:
        try:
            if await self._already_seen(video.get("url")):
                logger.info("[youtube_scout] Duplicate skipped: {}", video.get("url"))
                return None

            transcript = await self._get_transcript(video)
            transcript_unavailable = False

            if transcript:
                summary = await self._summarize_transcript(video.get("title", ""), transcript)
            else:
                transcript_unavailable = True
                logger.warning(
                    "[youtube_scout] No transcript; falling back to title/description for '{}'",
                    video.get("title"),
                )
                summary = (
                    "Transcript unavailable. Fallback source metadata only. "
                    f"Title: {video.get('title', '')}. "
                    f"Description: {video.get('description', '')}"
                )

            features_reference = await self._feature_reference()

            msg = self._claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                messages=[{
                    "role": "user",
                    "content": _EXTRACT_PROMPT.format(
                        title=video["title"],
                        summary=summary,
                        features_reference=features_reference,
                    ),
                }],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            data = json.loads(raw)
            video["transcript_summary"] = summary
            video["transcript_chars_used"] = len(transcript or "")
            video["transcript_unavailable"] = transcript_unavailable
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
