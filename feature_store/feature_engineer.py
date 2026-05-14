"""
ATLAS Feature Engineer
=======================
Computes ~50 technical / statistical features per symbol from the
market_data hypertable and batch-upserts them into the feature_store
hypertable every 60 seconds.

Feature groups
--------------
1.  Moving averages  – SMA 5/10/20/50/200, EMA 9/21
2.  RSI              – RSI-14, RSI-7 (short-term)
3.  MACD             – line, signal, histogram (12/26/9)
4.  Bollinger Bands  – upper/mid/lower/bandwidth/pct_b (20, 2σ)
5.  ATR              – ATR-14, normalised ATR-14
6.  VWAP             – session VWAP + deviation from VWAP
7.  Volume profile   – relative volume vs. 20-bar average
8.  Momentum         – ROC-10, ROC-5
9.  Volatility       – 20-bar rolling std (log returns), realised vol
10. Cross-asset      – 20-bar rolling correlation of close returns
                       (equities vs. BTC as crypto proxy)
11. Regime           – trend regime (price vs SMA-50), volatility regime,
                       golden/death cross signal

Run standalone:
    python -m feature_store.feature_engineer

All credentials loaded from config/keys.env via ATLASSettings.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from agents.base_agent import BaseAgent
from config.settings import settings
from database import connection as db

# ── Symbols ────────────────────────────────────────────────────────────────────

EQUITY_SYMBOLS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
CRYPTO_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_SYMBOLS:    list[str] = EQUITY_SYMBOLS + CRYPTO_SYMBOLS

# Bars needed for the longest indicator (SMA-200)
LOOKBACK_BARS = 220

# Feature schema version — bump when the feature set changes
FEATURE_VERSION = 1


# ── Data loader ────────────────────────────────────────────────────────────────


async def _load_ohlcv(symbol: str, limit: int = LOOKBACK_BARS) -> pd.DataFrame:
    """
    Fetch the most recent `limit` 1-minute OHLCV bars for `symbol`
    from market_data and return them as a time-sorted DataFrame.
    Returns an empty DataFrame when no data exists.
    """
    rows = await db.fetch(
        """
        SELECT timestamp, open, high, low, close, volume, vwap
          FROM market_data
         WHERE symbol = $1
         ORDER BY timestamp DESC
         LIMIT $2
        """,
        symbol,
        limit,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"],
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Feature computation ────────────────────────────────────────────────────────


def _compute_features(df: pd.DataFrame, symbol: str) -> dict[str, tuple[float, dict]]:
    """
    Compute all features for a single symbol given its OHLCV DataFrame.

    Returns:
        dict[feature_name, (value, meta_dict)]
        where meta_dict is stored in the feature_meta JSONB column.
    """
    if df.empty or len(df) < 5:
        return {}

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    n      = len(df)

    features: dict[str, tuple[float, dict]] = {}

    def _safe(val: Any, name: str, meta: dict | None = None) -> None:
        """Add feature only when value is finite."""
        try:
            v = float(val)
            if np.isfinite(v):
                features[name] = (v, meta or {})
        except (TypeError, ValueError):
            pass

    # ── 1. Simple Moving Averages ──────────────────────────────────────────────
    for period in (5, 10, 20, 50, 200):
        if n >= period:
            _safe(close.rolling(period).mean().iloc[-1],
                  f"sma_{period}", {"window": period, "type": "SMA"})

    # ── 2. Exponential Moving Averages ─────────────────────────────────────────
    for period in (9, 21):
        if n >= period:
            _safe(close.ewm(span=period, adjust=False).mean().iloc[-1],
                  f"ema_{period}", {"window": period, "type": "EMA"})

    # ── 3. RSI ─────────────────────────────────────────────────────────────────
    def _rsi(series: pd.Series, period: int) -> float | None:
        if len(series) <= period:
            return None
        delta = series.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    _safe(_rsi(close, 14), "rsi_14", {"window": 14, "type": "RSI"})
    _safe(_rsi(close, 7),  "rsi_7",  {"window": 7,  "type": "RSI"})

    # ── 4. MACD (12, 26, 9) ────────────────────────────────────────────────────
    if n >= 26:
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal
        meta_macd   = {"fast": 12, "slow": 26, "signal": 9, "type": "MACD"}
        _safe(macd_line.iloc[-1],   "macd_line",   meta_macd)
        _safe(macd_signal.iloc[-1], "macd_signal", meta_macd)
        _safe(macd_hist.iloc[-1],   "macd_hist",   meta_macd)

    # ── 5. Bollinger Bands (20, 2σ) ────────────────────────────────────────────
    if n >= 20:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std(ddof=0)
        bb_up  = bb_mid + 2 * bb_std
        bb_lo  = bb_mid - 2 * bb_std
        bb_bw  = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)   # bandwidth
        last_close = close.iloc[-1]
        denom      = (bb_up - bb_lo).replace(0, np.nan)
        bb_pct     = (last_close - bb_lo) / denom              # %B

        meta_bb = {"window": 20, "std_dev": 2, "type": "BB"}
        _safe(bb_up.iloc[-1],  "bb_upper",     meta_bb)
        _safe(bb_mid.iloc[-1], "bb_mid",        meta_bb)
        _safe(bb_lo.iloc[-1],  "bb_lower",      meta_bb)
        _safe(bb_bw.iloc[-1],  "bb_bandwidth",  meta_bb)
        _safe(bb_pct.iloc[-1], "bb_pct_b",      meta_bb)

    # ── 6. ATR-14 ─────────────────────────────────────────────────────────────
    if n >= 15:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
        _safe(atr14.iloc[-1], "atr_14", {"window": 14, "type": "ATR"})
        # Normalised ATR — ATR as % of close
        _safe(
            atr14.iloc[-1] / close.iloc[-1] * 100 if close.iloc[-1] != 0 else None,
            "atr_14_pct",
            {"window": 14, "type": "ATR_PCT"},
        )

    # ── 7. VWAP & VWAP deviation ───────────────────────────────────────────────
    # Use stored vwap column when available, else compute a rolling VWAP proxy
    stored_vwap = df["vwap"].dropna()
    if not stored_vwap.empty:
        vwap_val = float(stored_vwap.iloc[-1])
        _safe(vwap_val, "vwap", {"type": "VWAP", "source": "stored"})
        if close.iloc[-1] and vwap_val:
            _safe(
                (close.iloc[-1] - vwap_val) / vwap_val * 100,
                "vwap_dev_pct",
                {"type": "VWAP_DEV"},
            )
    else:
        # Proxy: cumulative (close * volume) / cumulative volume over lookback
        if volume.sum() > 0:
            typ_price = (high + low + close) / 3
            rolling_vwap = (typ_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            _safe(rolling_vwap.iloc[-1], "vwap", {"type": "VWAP", "source": "rolling20"})

    # ── 8. Volume profile — relative volume ─────────────────────────────────────
    if n >= 20:
        avg_vol = volume.rolling(20).mean().iloc[-1]
        if avg_vol and avg_vol > 0:
            _safe(
                volume.iloc[-1] / avg_vol,
                "rel_volume_20",
                {"window": 20, "type": "REL_VOL"},
            )
    if n >= 5:
        avg_vol5 = volume.rolling(5).mean().iloc[-1]
        if avg_vol5 and avg_vol5 > 0:
            _safe(
                volume.iloc[-1] / avg_vol5,
                "rel_volume_5",
                {"window": 5, "type": "REL_VOL"},
            )

    # ── 9. Momentum — Rate of Change ───────────────────────────────────────────
    for period in (5, 10):
        if n > period:
            prev = close.iloc[-(period + 1)]
            if prev and prev != 0:
                _safe(
                    (close.iloc[-1] - prev) / prev * 100,
                    f"roc_{period}",
                    {"window": period, "type": "ROC"},
                )

    # ── 10. Volatility ─────────────────────────────────────────────────────────
    if n >= 21:
        log_ret  = np.log(close / close.shift(1)).dropna()
        vol_20   = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252 * 390)  # annualised (1-min bars)
        _safe(vol_20, "vol_20_ann", {"window": 20, "type": "REALISED_VOL", "annualised": True})

        # Raw 20-bar std of close prices
        _safe(
            close.rolling(20).std(ddof=1).iloc[-1],
            "std_close_20",
            {"window": 20, "type": "STD_CLOSE"},
        )

    # ── 11. Regime indicators ──────────────────────────────────────────────────
    last_close_val = float(close.iloc[-1])

    # Trend regime: +1 = above SMA-50, -1 = below
    if n >= 50:
        sma50_val = float(close.rolling(50).mean().iloc[-1])
        _safe(
            1.0 if last_close_val > sma50_val else -1.0,
            "regime_trend",
            {"type": "TREND_REGIME", "reference": "SMA50"},
        )
        # Distance from SMA-50 as %
        if sma50_val != 0:
            _safe(
                (last_close_val - sma50_val) / sma50_val * 100,
                "dist_sma50_pct",
                {"type": "DIST_SMA50"},
            )

    # Golden / Death cross: SMA-50 vs SMA-200 → +1 golden, -1 death
    if n >= 200:
        sma50_val  = float(close.rolling(50).mean().iloc[-1])
        sma200_val = float(close.rolling(200).mean().iloc[-1])
        _safe(
            1.0 if sma50_val > sma200_val else -1.0,
            "golden_cross",
            {"type": "GOLDEN_CROSS", "fast": 50, "slow": 200},
        )

    # Volatility regime: high vol if annualised vol > 30%
    if "vol_20_ann" in features:
        _safe(
            1.0 if features["vol_20_ann"][0] > 0.30 else 0.0,
            "regime_high_vol",
            {"type": "VOL_REGIME", "threshold": 0.30},
        )

    return features


def _compute_cross_correlations(
    close_map: dict[str, pd.Series],
    window: int = 20,
) -> dict[str, dict[str, tuple[float, dict]]]:
    """
    Compute pairwise rolling correlation of log-returns for all symbol pairs
    present in close_map.  Returns a nested dict:
        symbol → { feature_name → (value, meta) }
    """
    result: dict[str, dict[str, tuple[float, dict]]] = {s: {} for s in close_map}

    symbols = list(close_map.keys())
    if len(symbols) < 2:
        return result

    # Align all series on a common index
    df_ret = pd.DataFrame(
        {sym: np.log(s / s.shift(1)) for sym, s in close_map.items()}
    ).dropna()

    if len(df_ret) < window:
        return result

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            try:
                corr = float(
                    df_ret[sym_a]
                    .rolling(window)
                    .corr(df_ret[sym_b])
                    .iloc[-1]
                )
                if not np.isfinite(corr):
                    continue
                feat_name = f"corr_{sym_b.lower()}_{window}"
                meta = {"type": "CORR", "pair": sym_b, "window": window}
                result[sym_a][feat_name] = (corr, meta)

                feat_name_b = f"corr_{sym_a.lower()}_{window}"
                result[sym_b][feat_name_b] = (corr, meta | {"pair": sym_a})
            except Exception:
                pass

    return result


# ── DB writer ──────────────────────────────────────────────────────────────────

_INSERT_SQL = """
    INSERT INTO feature_store
        (symbol, timestamp, feature_name, feature_value, feature_meta, version)
    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
    ON CONFLICT (symbol, timestamp, feature_name, version) DO UPDATE
        SET feature_value = EXCLUDED.feature_value,
            feature_meta  = EXCLUDED.feature_meta,
            computed_at   = NOW()
"""


async def _persist_features(
    symbol:    str,
    timestamp: datetime,
    features:  dict[str, tuple[float, dict]],
) -> None:
    """Batch-upsert all features for one symbol at one timestamp."""
    if not features:
        return

    rows = [
        (symbol, timestamp, name, value, json.dumps(meta), FEATURE_VERSION)
        for name, (value, meta) in features.items()
    ]

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(_INSERT_SQL, rows)

    logger.debug(
        "[feature_eng] {} | {} features saved @ {:%H:%M}",
        symbol, len(rows), timestamp,
    )


# ── FeatureEngineerAgent ───────────────────────────────────────────────────────


class FeatureEngineerAgent(BaseAgent):
    """
    Runs every 60 seconds, computes all features for every symbol,
    and bulk-upserts them into the feature_store hypertable.
    """

    agent_type = "signal"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(
            name="feature_engineer",
            config={"tick_seconds": 60, **(config or {})},
        )
        self._symbols = ALL_SYMBOLS
        logger.info("[feature_eng] Will process symbols: {}", self._symbols)

    async def setup(self) -> None:
        await db.init_pool()
        logger.info("[feature_eng] DB pool ready")

    async def run(self) -> None:
        """One full feature-computation cycle across all symbols."""
        cycle_start = datetime.now(tz=timezone.utc)
        logger.info("[feature_eng] ── Cycle start {:%H:%M:%S} UTC ──", cycle_start)

        # ── Load OHLCV for all symbols in parallel ─────────────────────────────
        tasks = {sym: asyncio.create_task(_load_ohlcv(sym)) for sym in self._symbols}
        ohlcv_map: dict[str, pd.DataFrame] = {}
        for sym, task in tasks.items():
            try:
                ohlcv_map[sym] = await task
            except Exception as exc:
                logger.warning("[feature_eng] Load failed for {}: {}", sym, exc)
                ohlcv_map[sym] = pd.DataFrame()

        # ── Cross-asset correlations (requires aligned close series) ───────────
        close_map: dict[str, pd.Series] = {}
        for sym, df in ohlcv_map.items():
            if not df.empty and len(df) >= 21:
                close_map[sym] = df.set_index("timestamp")["close"]

        cross_features = _compute_cross_correlations(close_map, window=20)

        # ── Per-symbol feature computation + persistence ───────────────────────
        total_features = 0
        for sym, df in ohlcv_map.items():
            if df.empty:
                logger.warning("[feature_eng] No data for {} — skipping", sym)
                continue

            # Timestamp anchor: latest bar's timestamp
            ts = df["timestamp"].iloc[-1]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            try:
                # Per-symbol technical features
                features = _compute_features(df, sym)

                # Merge in cross-asset correlations
                features.update(cross_features.get(sym, {}))

                await _persist_features(sym, ts, features)
                total_features += len(features)

                logger.info(
                    "[feature_eng] ✓ {} | {} features | close={:.4f}",
                    sym, len(features), float(df["close"].iloc[-1]),
                )

            except Exception as exc:
                logger.error(
                    "[feature_eng] Error computing features for {}: {}",
                    sym, exc, exc_info=True,
                )

        elapsed = (datetime.now(tz=timezone.utc) - cycle_start).total_seconds()
        logger.info(
            "[feature_eng] ── Cycle done in {:.2f}s | {} total features ──",
            elapsed, total_features,
        )

    async def teardown(self) -> None:
        await db.close_pool()
        logger.info("[feature_eng] Pool closed")


# ── Convenience function ───────────────────────────────────────────────────────


async def run_feature_engineer(symbols: list[str] | None = None) -> None:
    """
    Standalone coroutine: runs the FeatureEngineerAgent until cancelled.
    Useful when embedding in a larger asyncio application.
    """
    agent = FeatureEngineerAgent()
    if symbols:
        agent._symbols = symbols
    await agent.start()


# ── CLI ────────────────────────────────────────────────────────────────────────


async def _main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    loop     = asyncio.get_running_loop()
    stop_evt = asyncio.Event()

    def _shutdown(sig_name: str) -> None:
        logger.info("[feature_eng] {} — shutting down …", sig_name)
        stop_evt.set()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig.name)

    agent = FeatureEngineerAgent()
    task  = asyncio.create_task(agent.start())

    try:
        await asyncio.wait(
            [task, asyncio.create_task(stop_evt.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        await agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("[feature_eng] Stopped.")


if __name__ == "__main__":
    asyncio.run(_main())
