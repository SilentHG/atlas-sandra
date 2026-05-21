"""
ATLAS Pattern Recognition Engine — Day 5

Detects interpretable market patterns from OHLCV bars and stores them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import asyncpg
import pandas as pd


class PatternRecognitionEngine:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def detect_for_symbol(self, symbol: str, lookback_bars: int = 120) -> Dict[str, Any]:
        df = await self._load_data(symbol, lookback_bars)

        if df.empty or len(df) < 30:
            return {
                "symbol": symbol,
                "status": "insufficient_data",
                "bars_analysed": len(df),
                "patterns": [],
            }

        patterns: List[Dict[str, Any]] = []
        patterns += self._breakout(df)
        patterns += self._trend_continuation(df)
        patterns += self._volatility_squeeze(df)
        patterns += self._volume_anomaly(df)

        if not patterns:
            patterns.append(self._fallback_market_state(df))

        saved_ids = []
        for pattern in patterns:
            saved_ids.append(await self._save(symbol, pattern))

        return {
            "symbol": symbol,
            "status": "ok",
            "bars_analysed": len(df),
            "patterns_detected": len(patterns),
            "patterns": patterns,
            "saved_ids": saved_ids,
        }

    async def _load_data(self, symbol: str, lookback_bars: int) -> pd.DataFrame:
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                symbol,
                lookback_bars,
            )

        df = pd.DataFrame([dict(r) for r in rows])
        if df.empty:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()

    def _breakout(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        recent_close = float(df["close"].iloc[-1])
        prior_high = float(df["high"].iloc[-21:-1].max())

        if recent_close > prior_high:
            return [{
                "pattern": "20_bar_breakout",
                "description": "Latest close broke above the previous 20-bar high.",
                "confidence": 0.78,
                "supporting_data": {
                    "recent_close": recent_close,
                    "prior_high": prior_high,
                },
            }]
        return []

    def _trend_continuation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        close = df["close"].astype(float)
        ema_fast = close.ewm(span=10).mean().iloc[-1]
        ema_slow = close.ewm(span=30).mean().iloc[-1]

        if ema_fast > ema_slow and close.iloc[-1] > ema_fast:
            return [{
                "pattern": "bullish_trend_continuation",
                "description": "Fast EMA is above slow EMA and price is holding above the fast EMA.",
                "confidence": 0.72,
                "supporting_data": {
                    "ema_fast": round(float(ema_fast), 4),
                    "ema_slow": round(float(ema_slow), 4),
                    "close": round(float(close.iloc[-1]), 4),
                },
            }]
        return []

    def _volatility_squeeze(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        recent_range = float((df["high"] - df["low"]).tail(10).mean())
        longer_range = float((df["high"] - df["low"]).tail(50).mean())

        if longer_range > 0 and recent_range < longer_range * 0.55:
            return [{
                "pattern": "volatility_squeeze",
                "description": "Recent candle range has contracted sharply versus the longer baseline.",
                "confidence": 0.70,
                "supporting_data": {
                    "recent_avg_range": round(recent_range, 4),
                    "longer_avg_range": round(longer_range, 4),
                },
            }]
        return []

    def _volume_anomaly(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        recent_volume = float(df["volume"].iloc[-1])
        avg_volume = float(df["volume"].tail(50).mean())

        if avg_volume > 0 and recent_volume > avg_volume * 2:
            return [{
                "pattern": "volume_anomaly",
                "description": "Latest volume is more than 2x the recent average volume.",
                "confidence": 0.75,
                "supporting_data": {
                    "recent_volume": round(recent_volume, 4),
                    "avg_volume": round(avg_volume, 4),
                },
            }]
        return []

    def _fallback_market_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df["close"].astype(float)
        recent_close = float(close.iloc[-1])
        prior_close = float(close.iloc[max(0, len(close) - 21)])
        pct_change = ((recent_close - prior_close) / prior_close * 100) if prior_close else 0.0

        if pct_change > 1:
            pattern = "bullish_drift"
            description = "Price has drifted higher over the recent lookback."
        elif pct_change < -1:
            pattern = "bearish_drift"
            description = "Price has drifted lower over the recent lookback."
        else:
            pattern = "range_consolidation"
            description = "Price is consolidating without a strong directional breakout."

        return {
            "pattern": pattern,
            "description": description,
            "confidence": 0.60,
            "supporting_data": {
                "recent_close": round(recent_close, 4),
                "prior_close": round(prior_close, 4),
                "pct_change_20_bars": round(pct_change, 4),
            },
        }

    async def _save(self, symbol: str, pattern: Dict[str, Any]) -> str:
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO pattern_detections
                (symbol, pattern_name, description, confidence_score, supporting_data)
                VALUES ($1,$2,$3,$4,$5::jsonb)
                RETURNING id::text
                """,
                symbol,
                pattern["pattern"],
                pattern["description"],
                float(pattern["confidence"]),
                json.dumps(pattern.get("supporting_data", {})),
            )
