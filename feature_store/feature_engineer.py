"""
ATLAS Feature Engineer  ─ Day 2
================================
Computes 50+ technical features per symbol from the market_data
hypertable and batch-upserts them into feature_store every 60 seconds.

Feature groups
--------------
1.  SMA 5/10/20/50/200, EMA 9/21
2.  RSI-14, RSI-7
3.  MACD (12, 26, 9)
4.  Bollinger Bands (20, 2σ)
5.  ATR-14, normalised ATR
6.  VWAP + % deviation
7.  Relative volume (20-bar, 5-bar)
8.  Momentum / ROC-5, ROC-10
9.  Volatility-20 (annualised), std_close_20
10. Stochastic %K-14, %D-3
11. Williams %R-14
12. CCI-20
13. Cross-asset correlations (equity vs crypto, 20-bar)
14. Regime: trend, golden_cross, high_vol, ranging_score

Run standalone:
    python -m feature_store.feature_engineer
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from agents.base_agent import BaseAgent
from config.settings import settings
from database import connection as db

# ── Universe ───────────────────────────────────────────────────────────────────

EQUITY_SYMBOLS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
CRYPTO_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_SYMBOLS:    list[str] = EQUITY_SYMBOLS + CRYPTO_SYMBOLS
LOOKBACK_BARS               = 220
FEATURE_VERSION             = 2   # bumped: added Stochastic, W%R, CCI, ranging_score

# ── Data loader ────────────────────────────────────────────────────────────────


async def _load_ohlcv(symbol: str, limit: int = LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch most-recent `limit` bars for `symbol`, sorted ascending."""
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
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


async def _backfill_historical_data(symbol: str) -> None:
    """Fetch 5 days of 1-minute OHLCV from Polygon if no data exists."""
    count = await db.fetchval("SELECT COUNT(*) FROM market_data WHERE symbol = $1", symbol)
    if count and count > 0:
        return

    logger.info("[feature_eng] No data for {}. Backfilling last 5 days from Polygon...", symbol)
    
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=5)
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    ticker = symbol
    if symbol in CRYPTO_SYMBOLS:
        ticker = f"X:{symbol.replace('USDT', 'USD')}"
        
    api_key = settings.polygon_api_key
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            if not results:
                logger.warning("[feature_eng] Backfill returned no results for {}", symbol)
                return
                
            rows = []
            for r in results:
                ts = datetime.fromtimestamp(r["t"] / 1000.0, tz=timezone.utc)
                rows.append((
                    symbol, ts,
                    float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
                    float(r["v"]), float(r.get("vw", 0.0)),
                    int(r.get("n", 0)), "polygon_historical", "polygon_rest"
                ))
                
            _INSERT_SQL = """
                INSERT INTO market_data
                    (symbol, timestamp, open, high, low, close, volume, vwap, num_trades, exchange, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (symbol, timestamp) DO NOTHING
            """
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                await conn.executemany(_INSERT_SQL, rows)
                
            logger.success("[feature_eng] Backfilled {} rows for {}", len(rows), symbol)
    except Exception as exc:
        logger.error("[feature_eng] Failed to backfill {}: {}", symbol, exc)


# ── Feature computation ────────────────────────────────────────────────────────


def _compute_features(df: pd.DataFrame, symbol: str) -> dict[str, tuple[float, dict]]:
    """Return dict[feature_name → (value, meta)] for one symbol."""
    if df.empty or len(df) < 5:
        return {}

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    n      = len(df)

    features: dict[str, tuple[float, dict]] = {}

    def _safe(val: Any, name: str, meta: dict | None = None) -> None:
        try:
            v = float(val)
            if np.isfinite(v):
                features[name] = (v, meta or {})
        except (TypeError, ValueError):
            pass

    # 1. SMA
    for p in (5, 10, 20, 50, 200):
        if n >= p:
            _safe(close.rolling(p).mean().iloc[-1], f"sma_{p}", {"window": p, "type": "SMA"})

    # 2. EMA
    for p in (9, 21):
        if n >= p:
            _safe(close.ewm(span=p, adjust=False).mean().iloc[-1], f"ema_{p}", {"window": p, "type": "EMA"})

    # 3. RSI
    def _rsi(series: pd.Series, p: int) -> float | None:
        if len(series) <= p:
            return None
        d    = series.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
        rs   = gain / loss.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    _safe(_rsi(close, 14), "rsi_14", {"window": 14, "type": "RSI"})
    _safe(_rsi(close, 7),  "rsi_7",  {"window": 7,  "type": "RSI"})

    # 4. MACD (12, 26, 9)
    if n >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        ml    = ema12 - ema26
        ms    = ml.ewm(span=9, adjust=False).mean()
        mh    = ml - ms
        m_meta = {"fast": 12, "slow": 26, "signal": 9, "type": "MACD"}
        _safe(ml.iloc[-1], "macd_line",   m_meta)
        _safe(ms.iloc[-1], "macd_signal", m_meta)
        _safe(mh.iloc[-1], "macd_hist",   m_meta)

    # 5. Bollinger Bands (20, 2σ)
    if n >= 20:
        bm  = close.rolling(20).mean()
        bs  = close.rolling(20).std(ddof=0)
        bu  = bm + 2 * bs
        bl  = bm - 2 * bs
        bw  = (bu - bl) / bm.replace(0, np.nan)
        bp  = (close.iloc[-1] - bl) / (bu - bl).replace(0, np.nan)
        bb_m = {"window": 20, "std_dev": 2, "type": "BB"}
        _safe(bu.iloc[-1], "bb_upper",    bb_m)
        _safe(bm.iloc[-1], "bb_mid",       bb_m)
        _safe(bl.iloc[-1], "bb_lower",     bb_m)
        _safe(bw.iloc[-1], "bb_bandwidth", bb_m)
        _safe(bp.iloc[-1], "bb_pct_b",     bb_m)

    # 6. ATR-14
    if n >= 15:
        pc  = close.shift(1)
        tr  = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        a14 = tr.ewm(alpha=1/14, adjust=False).mean()
        _safe(a14.iloc[-1], "atr_14", {"window": 14, "type": "ATR"})
        if close.iloc[-1] != 0:
            _safe(a14.iloc[-1] / close.iloc[-1] * 100, "atr_14_pct", {"window": 14, "type": "ATR_PCT"})

    # 7. VWAP
    sv = df["vwap"].dropna()
    if not sv.empty:
        vv = float(sv.iloc[-1])
        _safe(vv, "vwap", {"type": "VWAP", "source": "stored"})
        if close.iloc[-1] and vv:
            _safe((close.iloc[-1] - vv) / vv * 100, "vwap_dev_pct", {"type": "VWAP_DEV"})
    elif volume.sum() > 0:
        tp = (high + low + close) / 3
        rv = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
        _safe(rv.iloc[-1], "vwap", {"type": "VWAP", "source": "rolling20"})

    # 8. Relative volume
    for w in (20, 5):
        if n >= w:
            avg = volume.rolling(w).mean().iloc[-1]
            if avg and avg > 0:
                _safe(volume.iloc[-1] / avg, f"rel_volume_{w}", {"window": w, "type": "REL_VOL"})

    # 9. Momentum (ROC)
    for p in (5, 10):
        if n > p:
            prev = close.iloc[-(p + 1)]
            if prev and prev != 0:
                _safe((close.iloc[-1] - prev) / prev * 100, f"roc_{p}", {"window": p, "type": "ROC"})

    # 10. Volatility
    if n >= 21:
        lr   = np.log(close / close.shift(1)).dropna()
        v20  = lr.rolling(20).std().iloc[-1] * np.sqrt(252 * 390)
        _safe(v20, "vol_20_ann", {"window": 20, "type": "REALISED_VOL", "annualised": True})
        _safe(close.rolling(20).std(ddof=1).iloc[-1], "std_close_20", {"window": 20, "type": "STD_CLOSE"})

    # 11. Stochastic Oscillator (%K-14, %D-3)
    if n >= 14:
        l14   = low.rolling(14).min()
        h14   = high.rolling(14).max()
        denom = (h14 - l14).replace(0, np.nan)
        k     = (close - l14) / denom * 100
        d     = k.rolling(3).mean()
        _safe(k.iloc[-1], "stoch_k_14", {"window": 14, "signal": 3, "type": "STOCH"})
        _safe(d.iloc[-1], "stoch_d_3",  {"window": 14, "signal": 3, "type": "STOCH_D"})

    # 12. Williams %R (14)
    if n >= 14:
        hi14  = high.rolling(14).max()
        lo14  = low.rolling(14).min()
        denom = (hi14 - lo14).replace(0, np.nan)
        wr    = (hi14 - close) / denom * -100
        _safe(wr.iloc[-1], "williams_r_14", {"window": 14, "type": "WILLIAMS_R"})

    # 13. CCI-20
    if n >= 20:
        tp      = (high + low + close) / 3
        tp_sma  = tp.rolling(20).mean()
        md      = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci     = (tp - tp_sma) / (0.015 * md.replace(0, np.nan))
        _safe(cci.iloc[-1], "cci_20", {"window": 20, "type": "CCI"})

    # 14. ADX-14
    if n >= 15:
        pc   = close.shift(1)
        ph   = high.shift(1)
        pl   = low.shift(1)
        plus_dm  = (high - ph).clip(lower=0).where((high - ph) > (pl - low), 0)
        minus_dm = (pl - low).clip(lower=0).where((pl - low) > (high - ph), 0)
        tr       = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        atr14    = tr.ewm(alpha=1/14, adjust=False).mean()
        pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx  = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        adx = dx.ewm(alpha=1/14, adjust=False).mean()
        _safe(adx.iloc[-1], "adx_14",  {"window": 14, "type": "ADX"})
        _safe(pdi.iloc[-1], "adx_pdi", {"window": 14, "type": "ADX_PDI"})
        _safe(mdi.iloc[-1], "adx_mdi", {"window": 14, "type": "ADX_MDI"})

    # 15. OBV (On Balance Volume)
    direction = np.sign(close.diff().fillna(0))
    obv = (volume * direction).cumsum()
    _safe(obv.iloc[-1], "obv", {"type": "OBV"})
    # OBV EMA signal (20-bar)
    if n >= 20:
        obv_ema = obv.ewm(span=20, adjust=False).mean()
        _safe(obv_ema.iloc[-1],             "obv_ema_20",  {"window": 20, "type": "OBV_EMA"})
        _safe(obv.iloc[-1] - obv_ema.iloc[-1], "obv_signal", {"type": "OBV_SIGNAL"})

    # 16. Price ROC-20
    if n > 20:
        prev20 = close.iloc[-(21)]
        if prev20 and prev20 != 0:
            _safe((close.iloc[-1] - prev20) / prev20 * 100, "roc_20", {"window": 20, "type": "ROC"})

    # 17. Volume SMA ratio (vol vs its own SMA-20)
    if n >= 20:
        vsma = volume.rolling(20).mean().iloc[-1]
        if vsma and vsma > 0:
            _safe(volume.iloc[-1] / vsma, "vol_sma_ratio_20", {"window": 20, "type": "VOL_SMA_RATIO"})

    # 18. High-Low range (absolute + % of close)
    hl_range = high.iloc[-1] - low.iloc[-1]
    _safe(hl_range, "hl_range", {"type": "HL_RANGE"})
    if close.iloc[-1] and close.iloc[-1] != 0:
        _safe(hl_range / close.iloc[-1] * 100, "hl_range_pct", {"type": "HL_RANGE_PCT"})

    # 19. Gap indicator (open vs prev close)
    if n >= 2:
        prev_close = close.iloc[-2]
        cur_open   = df["open"].iloc[-1]
        if prev_close and prev_close != 0:
            gap_pct = (cur_open - prev_close) / prev_close * 100
            _safe(gap_pct, "gap_pct", {"type": "GAP"})
            _safe(1.0 if abs(gap_pct) > 0.5 else 0.0, "gap_flag", {"type": "GAP_FLAG", "threshold": 0.5})

    # 20. Regime score (-3 to +3 composite)
    regime_pts = 0.0
    regime_cnt = 0
    if "regime_trend" in features:
        regime_pts += features["regime_trend"][0]; regime_cnt += 1
    if "golden_cross" in features:
        regime_pts += features["golden_cross"][0]; regime_cnt += 1
    if "rsi_14" in features:
        rsi_val = features["rsi_14"][0]
        regime_pts += (1.0 if rsi_val > 55 else -1.0 if rsi_val < 45 else 0.0)
        regime_cnt += 1
    if regime_cnt > 0:
        _safe(regime_pts, "regime_score", {"type": "REGIME_SCORE", "components": regime_cnt})

    # 21. BB position (already have bb_pct_b — add explicit band label)
    if "bb_pct_b" in features:
        bp_val = features["bb_pct_b"][0]
        band = 2.0 if bp_val > 1.0 else -2.0 if bp_val < 0.0 else (1.0 if bp_val > 0.5 else -1.0 if bp_val < 0.5 else 0.0)
        _safe(band, "bb_position", {"type": "BB_POSITION"})

    # Regime indicators (trend/golden cross/vol) — moved here from old #14
    lc = float(close.iloc[-1])
    if n >= 50:
        s50 = float(close.rolling(50).mean().iloc[-1])
        _safe(1.0 if lc > s50 else -1.0, "regime_trend", {"type": "TREND_REGIME", "ref": "SMA50"})
        if s50:
            _safe((lc - s50) / s50 * 100, "dist_sma50_pct", {"type": "DIST_SMA50"})

    if n >= 200:
        s50  = float(close.rolling(50).mean().iloc[-1])
        s200 = float(close.rolling(200).mean().iloc[-1])
        _safe(1.0 if s50 > s200 else -1.0, "golden_cross", {"type": "GOLDEN_CROSS", "fast": 50, "slow": 200})

    if "vol_20_ann" in features:
        _safe(1.0 if features["vol_20_ann"][0] > 0.30 else 0.0, "regime_high_vol",
              {"type": "VOL_REGIME", "threshold": 0.30})

    if n >= 20:
        s20  = close.rolling(20).mean().iloc[-1]
        st20 = close.rolling(20).std(ddof=1).iloc[-1]
        if s20 and s20 != 0 and st20:
            ranging = max(0.0, 1.0 - (st20 / s20) * 20)
            _safe(ranging, "ranging_score", {"window": 20, "type": "RANGING"})

    return features


def _compute_cross_correlations(
    close_map: dict[str, pd.Series],
    window: int = 20,
) -> dict[str, dict[str, tuple[float, dict]]]:
    """Return nested dict: symbol → {feature_name → (value, meta)}."""
    result: dict[str, dict[str, tuple[float, dict]]] = {s: {} for s in close_map}
    syms = list(close_map.keys())
    if len(syms) < 2:
        return result

    df_ret = pd.DataFrame(
        {s: np.log(v / v.shift(1)) for s, v in close_map.items()}
    ).dropna()
    if len(df_ret) < window:
        return result

    for i, sym_a in enumerate(syms):
        for sym_b in syms[i + 1:]:
            try:
                corr = float(df_ret[sym_a].rolling(window).corr(df_ret[sym_b]).iloc[-1])
                if not np.isfinite(corr):
                    continue
                meta = {"type": "CORR", "pair": sym_b, "window": window}
                result[sym_a][f"corr_{sym_b.lower()}_{window}"] = (corr, meta)
                result[sym_b][f"corr_{sym_a.lower()}_{window}"] = (corr, {**meta, "pair": sym_a})
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
    symbol: str, timestamp: datetime, features: dict[str, tuple[float, dict]]
) -> None:
    if not features:
        return
    rows = [
        (symbol, timestamp, name, value, json.dumps(meta), FEATURE_VERSION)
        for name, (value, meta) in features.items()
    ]
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(_INSERT_SQL, rows)
    logger.debug("[feature_eng] {} | {} features saved @ {:%H:%M}", symbol, len(rows), timestamp)


# ── Agent ─────────────────────────────────────────────────────────────────────


class FeatureEngineerAgent(BaseAgent):
    """Runs every 60 s and upserts 50+ features per symbol into feature_store."""

    agent_type = "signal"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="feature_engineer", config={"tick_seconds": 60, **(config or {})})
        self._symbols = ALL_SYMBOLS
        logger.info("[feature_eng] Tracking {} symbols", len(self._symbols))

    async def setup(self) -> None:
        await db.init_pool()
        logger.info("[feature_eng] DB pool ready")
        
        # Check and backfill missing data before starting cycle
        for sym in self._symbols:
            await _backfill_historical_data(sym)

    async def run(self) -> None:
        t0 = datetime.now(tz=timezone.utc)
        logger.info("[feature_eng] ── Cycle {:%H:%M:%S} UTC ──", t0)

        tasks    = {s: asyncio.create_task(_load_ohlcv(s)) for s in self._symbols}
        ohlcv    = {}
        for sym, task in tasks.items():
            try:
                ohlcv[sym] = await task
            except Exception as exc:
                logger.warning("[feature_eng] Load failed {}: {}", sym, exc)
                ohlcv[sym] = pd.DataFrame()

        close_map = {s: df.set_index("timestamp")["close"]
                     for s, df in ohlcv.items() if not df.empty and len(df) >= 21}
        cross = _compute_cross_correlations(close_map)

        total = 0
        for sym, df in ohlcv.items():
            if df.empty:
                logger.warning("[feature_eng] No data for {} — skip", sym)
                continue
            ts = df["timestamp"].iloc[-1]
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            try:
                feats = _compute_features(df, sym)
                feats.update(cross.get(sym, {}))
                await _persist_features(sym, ts, feats)
                total += len(feats)
                logger.info("[feature_eng] ✓ {} | {} feats | close={:.4f}",
                            sym, len(feats), float(df["close"].iloc[-1]))
            except Exception as exc:
                logger.error("[feature_eng] Error {}: {}", sym, exc, exc_info=True)

        logger.info("[feature_eng] ── Done {:.2f}s | {} total ──",
                    (datetime.now(tz=timezone.utc) - t0).total_seconds(), total)

    async def teardown(self) -> None:
        await db.close_pool()
        logger.info("[feature_eng] Pool closed")


# ── Convenience / CLI ─────────────────────────────────────────────────────────


async def run_feature_engineer(symbols: list[str] | None = None) -> None:
    agent = FeatureEngineerAgent()
    if symbols:
        agent._symbols = symbols
    await agent.start()


async def _main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True,
               format=("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level:<8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"))
    stop_evt = asyncio.Event()
    loop     = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_evt.set)

    agent = FeatureEngineerAgent()
    task  = asyncio.create_task(agent.start())
    try:
        await asyncio.wait([task, asyncio.create_task(stop_evt.wait())],
                           return_when=asyncio.FIRST_COMPLETED)
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
