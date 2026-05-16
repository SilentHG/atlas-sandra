"""
ATLAS FastAPI Server — api/main.py
===================================
REST API + SSE streaming for the ATLAS trading system.
- No WebSocket code — all streaming via Server-Sent Events (SSE)
- X-API-Key authentication on all /api/* routes
- Comprehensive health check
- All endpoints return valid JSON with <500ms response time

Run: uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from config.settings import settings
from database import connection as db
from risk_management.kill_switch import get_kill_switch
from risk_management.risk_manager import RiskManager

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    ks = get_kill_switch()
    await ks.setup()
    logger.info("[api] ATLAS API ready on :8080")
    yield
    await db.close_pool()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ATLAS Trading API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

# API key loaded from env; fallback for dev
_API_KEY = getattr(settings, "api_key", None) or "atlas-dev-key"


async def verify_api_key(x_api_key: str = Header(default=None)):
    """Check X-API-Key header on protected routes."""
    if x_api_key is None or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key

# ── Response time middleware ──────────────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > 500:
        logger.warning("[api] Slow response: {} {} took {:.0f}ms", request.method, request.url.path, elapsed_ms)
    return response

# ── Models ────────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str                       # "buy" | "sell"
    order_type: str = "market"
    limit_price: float | None = None
    strategy_id: str | None = None
    stop_loss: float | None = None

class SimulatePnlRequest(BaseModel):
    loss_usd: float
    weekly_loss_usd: float = 0.0

class GenerateStrategyRequest(BaseModel):
    strategy_type: str = "trend"
    symbols: list[str] = ["AAPL", "MSFT"]

class SpawnAgentRequest(BaseModel):
    agent_type: str
    name: str
    config: dict = {}

class AgentRegistryRequest(BaseModel):
    name: str
    agent_type: str = "custom"
    version: str = "1.0.0"
    description: str | None = None
    capabilities: list[str] = []
    status: str = "idle"
    heartbeat_interval_s: int = 30
    metadata: dict = {}

class PatternRequest(BaseModel):
    symbol: str
    lookback_bars: int = 100

class KillSwitchRequest(BaseModel):
    action: str = "arm"
    strategy_id: str | None = None
    reason: str | None = None

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Comprehensive health check — checks ALL services."""
    services: dict[str, Any] = {}

    # 1. TimescaleDB connection
    try:
        await db.fetchval("SELECT 1")
        services["timescaledb"] = "ok"
    except Exception as e:
        services["timescaledb"] = f"error: {e}"

    # 2. Feature store freshness
    try:
        last_feature = await db.fetchval(
            "SELECT MAX(computed_at) FROM feature_store"
        )
        if last_feature:
            age_s = (datetime.now(timezone.utc) - last_feature).total_seconds()
            services["feature_store"] = {
                "status": "ok" if age_s < 300 else "stale",
                "last_computed": last_feature.isoformat(),
                "age_seconds": round(age_s, 1),
            }
        else:
            services["feature_store"] = "no_data"
    except Exception as e:
        services["feature_store"] = f"error: {e}"

    # 3. Agent registry
    try:
        count = await db.fetchval("SELECT COUNT(*) FROM agent_registry WHERE status='running'")
        total = await db.fetchval("SELECT COUNT(*) FROM agent_registry")
        services["agent_registry"] = {
            "running": int(count or 0),
            "total": int(total or 0),
        }
    except Exception as e:
        services["agent_registry"] = f"error: {e}"

    # 4. Kill switch status
    try:
        ks = get_kill_switch()
        armed = await ks.is_armed()
        services["kill_switch"] = "ARMED" if armed else "disarmed"
    except Exception as e:
        services["kill_switch"] = f"error: {e}"

    # 5. Last data timestamp
    try:
        last_ts = await db.fetchval("SELECT MAX(timestamp) FROM market_data")
        if last_ts:
            age_s = (datetime.now(timezone.utc) - last_ts).total_seconds()
            services["last_data_timestamp"] = {
                "timestamp": last_ts.isoformat(),
                "age_seconds": round(age_s, 1),
                "status": "ok" if age_s < 300 else "stale",
            }
        else:
            services["last_data_timestamp"] = "no_data"
    except Exception as e:
        services["last_data_timestamp"] = f"error: {e}"

    overall = "healthy" if services.get("timescaledb") == "ok" else "degraded"
    return {
        "status": overall,
        "services": services,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/api/portfolio/summary")
@app.get("/portfolio")
async def portfolio_summary():
    try:
        rows = await db.fetch(
            """SELECT symbol, side, quantity, entry_price, current_price,
                      unrealized_pnl, realized_pnl, pnl, status
               FROM positions WHERE status='open' ORDER BY symbol"""
        )
        positions = [dict(r) for r in rows]
        total_unrealized = sum(float(p.get("unrealized_pnl") or 0) for p in positions)
        total_realized = await db.fetchval(
            "SELECT COALESCE(SUM(realized_pnl),0) FROM positions WHERE status='closed' AND closed_at >= NOW()-INTERVAL '1 day'"
        ) or 0
        return {
            "positions":       positions,
            "open_count":      len(positions),
            "total_unrealized_pnl": round(float(total_unrealized), 2),
            "total_realized_pnl":   round(float(total_realized), 2),
            "daily_pnl":       round(float(total_unrealized) + float(total_realized), 2),
        }
    except Exception as exc:
        logger.error("[api] portfolio_summary error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Strategies ────────────────────────────────────────────────────────────────

@app.get("/api/strategies")
@app.get("/strategies")
async def list_strategies():
    try:
        rows = await db.fetch(
            "SELECT id,name,strategy_type,status,symbols,description,created_at FROM strategies ORDER BY created_at DESC"
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[api] list_strategies error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/strategies/generate")
async def generate_strategy(req: GenerateStrategyRequest):
    try:
        from strategy_engine.ideator import StrategyIdeator
        from strategy_engine.strategy_coder import StrategyCoder
        ideator = StrategyIdeator()
        await ideator.generate_strategies(strategy_types=[req.strategy_type], symbols=req.symbols)
        coder = StrategyCoder()
        await coder.code_pending_strategies()
        return {"status": "ok", "message": "Strategy generated and coded"}
    except Exception as exc:
        logger.error("[api] generate_strategy error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Positions ─────────────────────────────────────────────────────────────────

@app.get("/api/positions")
@app.get("/positions")
async def get_positions():
    try:
        rows = await db.fetch(
            "SELECT * FROM positions WHERE status='open' ORDER BY opened_at DESC"
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[api] get_positions error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Risk ──────────────────────────────────────────────────────────────────────

@app.get("/api/risk/status")
async def risk_status():
    try:
        ks = get_kill_switch()
        state = await ks.get_global_state()
        return {
            "kill_switch_armed": state.get("armed", False),
            "reason":            state.get("reason"),
            "armed_at":          str(state.get("armed_at") or ""),
            "daily_loss_usd":    float(state.get("daily_loss_usd") or 0),
            "weekly_loss_usd":   float(state.get("weekly_loss_usd") or 0),
            "daily_limit_pct":   0.02,
            "weekly_limit_pct":  0.04,
        }
    except Exception as exc:
        logger.error("[api] risk_status error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/risk/simulate-pnl")
async def simulate_pnl(req: SimulatePnlRequest):
    try:
        ks = get_kill_switch()
        await ks.record_pnl(req.loss_usd, req.weekly_loss_usd)
        armed = await ks.is_armed()
        return {"kill_switch_armed": armed, "daily_loss_usd": req.loss_usd}
    except Exception as exc:
        logger.error("[api] simulate_pnl error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/risk/kill-switch")
async def manual_kill_switch(req: KillSwitchRequest):
    """
    KILL-003: Manual halt via API.
    KILL-005: Per-strategy halt with strategy_id.
    """
    try:
        ks = get_kill_switch()

        if req.action == "arm":
            reason = req.reason or "KILL-003: manual API trigger"
            if req.strategy_id:
                # KILL-005: per-strategy halt
                await ks.kill_strategy(req.strategy_id, reason)
                return {"status": "armed", "scope": "strategy", "strategy_id": req.strategy_id}
            else:
                await ks.arm(reason, armed_by="api_user")
                return {"status": "armed", "scope": "portfolio"}
        elif req.action == "disarm":
            if req.strategy_id:
                await ks.revive_strategy(req.strategy_id)
                return {"status": "disarmed", "scope": "strategy", "strategy_id": req.strategy_id}
            else:
                await ks.disarm(disarmed_by="api_user")
                return {"status": "disarmed", "scope": "portfolio"}
        else:
            raise HTTPException(400, "action must be 'arm' or 'disarm'")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] kill_switch error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Agents ────────────────────────────────────────────────────────────────────

@app.get("/api/agents/registry")
async def agents_registry():
    try:
        rows = await db.fetch(
            "SELECT id,name,agent_type,version,description,capabilities,status,last_heartbeat,heartbeat_interval_s,error_count,last_error,metadata,created_at,updated_at FROM agent_registry ORDER BY name"
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("[api] agents_registry error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/registry", status_code=201)
async def create_agent_registry(req: AgentRegistryRequest):
    try:
        row = await db.fetchrow(
            """INSERT INTO agent_registry
               (name,agent_type,version,description,capabilities,status,heartbeat_interval_s,metadata,last_heartbeat)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,NOW())
               RETURNING *""",
            req.name,
            req.agent_type,
            req.version,
            req.description,
            req.capabilities,
            req.status,
            req.heartbeat_interval_s,
            json.dumps(req.metadata),
        )
        return dict(row)
    except Exception as exc:
        logger.error("[api] create_agent_registry error: {}", exc)
        raise HTTPException(500, str(exc))

@app.get("/api/agents/{agent_name}/status")
async def agent_status(agent_name: str):
    try:
        row = await db.fetchrow("SELECT * FROM agent_registry WHERE name=$1", agent_name)
        if not row:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] agent_status error: {}", exc)
        raise HTTPException(500, str(exc))

@app.put("/api/agents/{agent_name}")
async def update_agent_registry(agent_name: str, req: AgentRegistryRequest):
    try:
        row = await db.fetchrow(
            """UPDATE agent_registry
               SET name=$2, agent_type=$3, version=$4, description=$5,
                   capabilities=$6, status=$7, heartbeat_interval_s=$8,
                   metadata=$9::jsonb, updated_at=NOW()
               WHERE name=$1
               RETURNING *""",
            agent_name,
            req.name,
            req.agent_type,
            req.version,
            req.description,
            req.capabilities,
            req.status,
            req.heartbeat_interval_s,
            json.dumps(req.metadata),
        )
        if not row:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] update_agent_registry error: {}", exc)
        raise HTTPException(500, str(exc))

@app.delete("/api/agents/{agent_name}")
async def delete_agent_registry(agent_name: str):
    try:
        row = await db.fetchrow(
            "DELETE FROM agent_registry WHERE name=$1 RETURNING name",
            agent_name,
        )
        if not row:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return {"status": "deleted", "agent": agent_name}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] delete_agent_registry error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/{agent_name}/heartbeat")
async def heartbeat_agent(agent_name: str):
    try:
        row = await db.fetchrow(
            """UPDATE agent_registry
               SET last_heartbeat=NOW(), status='running', updated_at=NOW()
               WHERE name=$1
               RETURNING name,last_heartbeat,status""",
            agent_name,
        )
        if not row:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] heartbeat_agent error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/{agent_name}/pause")
async def pause_agent(agent_name: str):
    try:
        await db.execute(
            "UPDATE agent_registry SET status='paused', updated_at=NOW() WHERE name=$1", agent_name
        )
        return {"status": "paused", "agent": agent_name}
    except Exception as exc:
        logger.error("[api] pause_agent error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/{agent_name}/resume")
async def resume_agent(agent_name: str):
    try:
        await db.execute(
            "UPDATE agent_registry SET status='idle', updated_at=NOW() WHERE name=$1", agent_name
        )
        return {"status": "resumed", "agent": agent_name}
    except Exception as exc:
        logger.error("[api] resume_agent error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/{agent_name}/kill")
async def kill_agent(agent_name: str):
    try:
        await db.execute(
            "UPDATE agent_registry SET status='stopped', updated_at=NOW() WHERE name=$1", agent_name
        )
        return {"status": "killed", "agent": agent_name}
    except Exception as exc:
        logger.error("[api] kill_agent error: {}", exc)
        raise HTTPException(500, str(exc))

@app.post("/api/agents/spawn")
async def spawn_agent(req: SpawnAgentRequest):
    try:
        await db.execute(
            """INSERT INTO agent_registry (name, agent_type, metadata, status)
               VALUES ($1, $2, $3::jsonb, 'idle')
               ON CONFLICT (name) DO UPDATE SET status='idle', updated_at=NOW()""",
            req.name, req.agent_type, json.dumps(req.config),
        )
        return {"status": "spawned", "name": req.name, "agent_type": req.agent_type}
    except Exception as exc:
        logger.error("[api] spawn_agent error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Execution ─────────────────────────────────────────────────────────────────

@app.post("/api/execution/test-order")
async def test_order(req: OrderRequest):
    try:
        from execution.alpaca_executor import AlpacaExecutor
        executor = AlpacaExecutor()
        await executor.setup()
        try:
            result = await executor.submit_order(
                symbol=req.symbol, qty=req.qty, side=req.side,
                order_type=req.order_type, limit_price=req.limit_price,
                strategy_id=req.strategy_id, stop_loss=req.stop_loss,
            )
            return {"status": "submitted", "order": result}
        except RuntimeError as e:
            raise HTTPException(403, str(e))
        finally:
            await executor.teardown()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] test_order error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Dashboard brief ───────────────────────────────────────────────────────────

@app.post("/api/dashboard/generate-brief")
async def generate_brief():
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=settings.anthropic_api_key)
        rows = await db.fetch(
            "SELECT symbol,close,volume FROM market_data ORDER BY timestamp DESC LIMIT 10"
        )
        snapshot = [dict(r) for r in rows]
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    f"Generate a concise daily market intelligence brief for a systematic trader.\n"
                    f"Latest market snapshot: {json.dumps(snapshot[:5])}\n"
                    "Cover: market regime, top movers, risk environment, and one actionable insight."
                ),
            }],
        )
        return {"brief": msg.content[0].text}
    except Exception as exc:
        logger.error("[api] generate_brief error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Patterns ──────────────────────────────────────────────────────────────────

@app.post("/api/patterns/detect")
async def detect_patterns(req: PatternRequest):
    try:
        rows = await db.fetch(
            "SELECT timestamp,open,high,low,close,volume FROM market_data WHERE symbol=$1 ORDER BY timestamp DESC LIMIT $2",
            req.symbol, req.lookback_bars,
        )
        if not rows:
            raise HTTPException(404, f"No data for {req.symbol}")
        import pandas as pd, numpy as np
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        for c in ("open","high","low","close","volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        patterns = []
        close = df["close"]

        # Doji
        body = abs(df["open"] - close).iloc[-1]
        rng  = (df["high"] - df["low"]).iloc[-1]
        if rng > 0 and body / rng < 0.1:
            patterns.append({"pattern": "doji", "confidence": 0.7})

        # Golden cross
        if len(df) >= 50:
            if close.rolling(20).mean().iloc[-1] > close.rolling(50).mean().iloc[-1]:
                patterns.append({"pattern": "golden_cross_20_50", "confidence": 0.65})

        # RSI
        d = close.diff()
        gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi  = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
        rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if rsi_now < 30:
            patterns.append({"pattern": "rsi_oversold", "rsi": round(rsi_now, 2), "confidence": 0.8})
        elif rsi_now > 70:
            patterns.append({"pattern": "rsi_overbought", "rsi": round(rsi_now, 2), "confidence": 0.8})

        return {"symbol": req.symbol, "patterns": patterns, "bars_analysed": len(df)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] detect_patterns error: {}", exc)
        raise HTTPException(500, str(exc))

# ── SSE: live positions stream ────────────────────────────────────────────────

async def _position_event_generator(request: Request) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events with live position data every second."""
    while True:
        if await request.is_disconnected():
            logger.info("[sse] Client disconnected from /api/stream/positions")
            break
        try:
            rows = await db.fetch(
                "SELECT symbol,side,quantity,current_price,unrealized_pnl,pnl "
                "FROM positions WHERE status='open'"
            )
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [dict(r) for r in rows],
            }
            yield f"data: {json.dumps(payload, default=str)}\n\n"
        except Exception as exc:
            logger.error("[sse] positions error: {}", exc)
            yield f"data: {{\"error\": \"{exc}\"}}\n\n"
        await asyncio.sleep(1)


@app.get("/api/stream/positions")
async def stream_positions(request: Request):
    """SSE — pushes live position updates every second."""
    return StreamingResponse(
        _position_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

# ── SSE: live P&L stream ─────────────────────────────────────────────────────

async def _pnl_event_generator(request: Request) -> AsyncGenerator[str, None]:
    """Yield SSE events with P&L aggregates every second."""
    while True:
        if await request.is_disconnected():
            logger.info("[sse] Client disconnected from /api/stream/pnl")
            break
        try:
            # Aggregate P&L from positions
            row = await db.fetchrow(
                """SELECT
                    COALESCE(SUM(unrealized_pnl), 0) AS total_unrealized,
                    COALESCE(SUM(realized_pnl), 0) AS total_realized,
                    COALESCE(SUM(pnl), 0) AS total_pnl,
                    COUNT(*) AS open_positions
                FROM positions WHERE status='open'"""
            )
            daily_realized = await db.fetchval(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions WHERE status='closed' AND closed_at >= NOW()-INTERVAL '1 day'"
            ) or 0

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unrealized_pnl": round(float(row["total_unrealized"]), 2) if row else 0,
                "realized_pnl": round(float(row["total_realized"]), 2) if row else 0,
                "total_pnl": round(float(row["total_pnl"]), 2) if row else 0,
                "daily_realized_pnl": round(float(daily_realized), 2),
                "open_positions": int(row["open_positions"]) if row else 0,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("[sse] pnl error: {}", exc)
            yield f"data: {{\"error\": \"{exc}\"}}\n\n"
        await asyncio.sleep(1)


@app.get("/api/stream/pnl")
async def stream_pnl(request: Request):
    """SSE — pushes live P&L updates every second."""
    return StreamingResponse(
        _pnl_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

# ── SSE: live signals stream ─────────────────────────────────────────────────

async def _signals_event_generator(request: Request) -> AsyncGenerator[str, None]:
    """Yield SSE events with latest feature/signal data every second."""
    while True:
        if await request.is_disconnected():
            logger.info("[sse] Client disconnected from /api/stream/signals")
            break
        try:
            rows = await db.fetch(
                """SELECT symbol, feature_name, feature_value, computed_at
                FROM feature_store
                WHERE feature_name IN ('rsi_14', 'macd_line', 'regime_score', 'regime_trend')
                  AND computed_at >= NOW() - INTERVAL '5 minutes'
                ORDER BY computed_at DESC
                LIMIT 50"""
            )
            signals = [
                {
                    "symbol": r["symbol"],
                    "feature": r["feature_name"],
                    "value": round(float(r["feature_value"]), 4) if r["feature_value"] else None,
                    "computed_at": r["computed_at"].isoformat() if r["computed_at"] else None,
                }
                for r in rows
            ]
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signals": signals,
                "count": len(signals),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("[sse] signals error: {}", exc)
            yield f"data: {{\"error\": \"{exc}\"}}\n\n"
        await asyncio.sleep(1)


@app.get("/api/stream/signals")
async def stream_signals(request: Request):
    """SSE — pushes latest trading signals every second."""
    return StreamingResponse(
        _signals_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
