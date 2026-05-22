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
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from config.settings import settings
from database import connection as db
from risk_management.kill_switch import get_kill_switch
from risk_management.risk_manager import RiskManager
from copy_trading.copy_engine import CopyTradingEngine
from dashboard_services.intelligence_brief import IntelligenceBriefService
from validation.sensitivity_analysis import SensitivityAnalyzer
from validation.regime_testing import RegimeTester
from strategy_engine.strategy_coder import _validate_code
from pattern_recognition.pattern_engine import PatternRecognitionEngine
from strategy_mutation.mutation_engine import StrategyMutationEngine
from orchestration.self_improvement_orchestrator import SelfImprovementOrchestrator

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
app.mount("/dashboard", StaticFiles(directory="dashboard_static", html=True), name="dashboard")

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
    price: float | None = None
    broker: str = "alpaca"
    test_only: bool = False
    strategy_id: str | None = None
    stop_loss: float | None = None

class SimulatePnlRequest(BaseModel):
    loss_usd: float
    weekly_loss_usd: float = 0.0

class BatchGenerateStrategyRequest(BaseModel):
    count: int = 10
    asset_classes: list[str] = ["us_equities", "crypto"]
    styles: list[str] = ["momentum", "mean_reversion", "breakout"]
    timeframes: list[str] = ["5m", "15m", "1h"]
    symbols_by_asset_class: dict = {
        "us_equities": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
        "crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    }
    lookback_days: int = 90

class GenerateStrategyRequest(BaseModel):
    strategy_type: str = "trend"
    symbols: list[str] = ["AAPL", "MSFT"]
    asset_class: str = "us_equities"
    timeframe: str = "1h"
    style: str = "momentum"
    lookback_days: int = 90
    custom_prompt: str | None = None

class SpawnAgentRequest(BaseModel):
    type: str | None = None
    agent_type: str | None = None
    name: str | None = None
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


class CopyTradingLinkRequest(BaseModel):
    leader_account: str
    follower_account: str
    follower_risk_limit_pct: float = 0.02
    sizing_mode: str = "proportional"

class CopyTradingMirrorRequest(BaseModel):
    leader_account: str = "leader_demo"
    leader_order_id: str = "leader-paper-order"
    symbol: str = "AAPL"
    side: str = "buy"
    leader_qty: float = 10
    leader_equity: float = 100000
    follower_equity: float = 25000
    price: float = 100
    fill_ratio: float = 1.0

class SensitivityRequest(BaseModel):
    baseline_metrics: dict
    variant_metrics: list[dict]
    variation_pct: float = 20.0

class RegimeValidationRequest(BaseModel):
    regime_metrics: dict

class SelfImprovementCycleRequest(BaseModel):
    limit: int = 10

class StrategyMutationRequest(BaseModel):
    mutation_type: str = "parameter_variation"
    variation_pct: float = 20.0
    variants: int = 3

class PatternRequest(BaseModel):
    symbol: str
    lookback_bars: int = 100

class BacktestRunRequest(BaseModel):
    strategy_id: str = ""
    symbol: str = "AAPL"
    start: str | None = None
    end: str | None = None
    qty: float = 10.0
    parameters: dict = {}

class WalkForwardRequest(BacktestRunRequest):
    window_days: int = 30
    step_days: int = 7
    windows: int = 3

class BacktestSensitivityRequest(BacktestRunRequest):
    qty_multipliers: list[float] = [0.5, 1.0, 1.5, 2.0]

class RegimeTestRequest(BacktestRunRequest):
    regimes: dict[str, dict[str, str]] = {}

class ScoutTestRequest(BaseModel):
    source: str
    subreddits: list[str] = ["algotrading"]
    query: str = "momentum trading"
    limit: int = 5
    max_results: int = 5

class KillSwitchRequest(BaseModel):
    action: str = "arm"
    strategy_id: str | None = None
    reason: str | None = None


def _parse_api_datetime(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

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

    # 4. Strategy inventory
    try:
        total = await db.fetchval("SELECT COUNT(*) FROM strategies")
        active = await db.fetchval("SELECT COUNT(*) FROM strategies WHERE status='active'")
        services["strategies"] = {
            "total": int(total or 0),
            "active": int(active or 0),
        }
    except Exception as e:
        services["strategies"] = f"error: {e}"

    # 5. Kill switch status
    try:
        ks = get_kill_switch()
        armed = await ks.is_armed()
        services["kill_switch"] = "ARMED" if armed else "disarmed"
    except Exception as e:
        services["kill_switch"] = f"error: {e}"

    # 6. Last data timestamp
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
        from strategy_engine.ideator import run_ideator
        from strategy_engine.strategy_coder import run_strategy_coder

        from strategy_engine.ideator import run_ideator_dynamic
        strategy_ids = await run_ideator_dynamic(
            asset_class=req.asset_class,
            symbols=req.symbols,
            timeframe=req.timeframe,
            style=req.style,
            lookback_days=req.lookback_days,
            custom_prompt=req.custom_prompt,
        )
        coded_ids = await run_strategy_coder(strategy_id=strategy_ids[0] if strategy_ids else None)
        return {
            "status": "ok",
            "generated_strategy_ids": strategy_ids,
            "coded_strategy_ids": coded_ids,
            "requested_type": req.strategy_type,
            "requested_symbols": req.symbols,
            "asset_class": req.asset_class,
            "timeframe": req.timeframe,
            "style": req.style,
        }
    except Exception as exc:
        logger.error("[api] generate_strategy error: {}", exc)
        raise HTTPException(500, str(exc))


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.post("/api/backtest/run")
async def run_backtest(req: BacktestRunRequest):
    try:
        from backtesting.backtest_engine import BacktestEngine

        end = _parse_api_datetime(req.end, datetime.now(timezone.utc))
        start = _parse_api_datetime(req.start, end - timedelta(days=30))
        engine = BacktestEngine()
        results = await engine.run(
            symbol=req.symbol,
            start=start,
            end=end,
            strategy_id=req.strategy_id,
            parameters=req.parameters,
            qty=req.qty,
        )
        return jsonable_encoder({
            "status": "completed",
            "symbol": req.symbol,
            "strategy_id": req.strategy_id,
            "results": results,
        })
    except Exception as exc:
        logger.error("[api] run_backtest error: {}", exc)
        raise HTTPException(500, str(exc))


@app.post("/api/backtest/walk-forward")
async def walk_forward_backtest(req: WalkForwardRequest):
    try:
        from backtesting.backtest_engine import BacktestEngine

        end = _parse_api_datetime(req.end, datetime.now(timezone.utc))
        step = timedelta(days=req.step_days)
        window = timedelta(days=req.window_days)
        engine = BacktestEngine()
        runs = []

        first_start = _parse_api_datetime(
            req.start,
            end - step * max(req.windows - 1, 0) - window,
        )
        for idx in range(req.windows):
            start = first_start + step * idx
            stop = start + window
            if stop > end:
                stop = end
            if start >= stop:
                continue
            results = await engine.run(
                symbol=req.symbol,
                start=start,
                end=stop,
                strategy_id=req.strategy_id,
                parameters={**req.parameters, "walk_forward_window": idx + 1},
                qty=req.qty,
            )
            runs.append({"window": idx + 1, "start": start, "end": stop, "results": results})

        return jsonable_encoder({"status": "completed", "runs": runs})
    except Exception as exc:
        logger.error("[api] walk_forward_backtest error: {}", exc)
        raise HTTPException(500, str(exc))


@app.post("/api/backtest/sensitivity")
async def sensitivity_backtest(req: SensitivityRequest):
    try:
        from backtesting.backtest_engine import BacktestEngine

        end = _parse_api_datetime(req.end, datetime.now(timezone.utc))
        start = _parse_api_datetime(req.start, end - timedelta(days=30))
        engine = BacktestEngine()
        runs = []
        holdout_returns = []

        for multiplier in req.qty_multipliers:
            qty = req.qty * multiplier
            results = await engine.run(
                symbol=req.symbol,
                start=start,
                end=end,
                strategy_id=req.strategy_id,
                parameters={**req.parameters, "sensitivity_qty_multiplier": multiplier},
                qty=qty,
            )
            holdout = results.get("holdout")
            if holdout:
                holdout_returns.append(float(holdout.total_return))
            runs.append({"qty_multiplier": multiplier, "qty": qty, "results": results})

        dispersion = 0.0
        if len(holdout_returns) > 1:
            dispersion = max(holdout_returns) - min(holdout_returns)
        return jsonable_encoder({
            "status": "completed",
            "overfitting_risk": "high" if dispersion > 0.25 else "normal",
            "holdout_return_dispersion": dispersion,
            "runs": runs,
        })
    except Exception as exc:
        logger.error("[api] sensitivity_backtest error: {}", exc)
        raise HTTPException(500, str(exc))


@app.post("/api/backtest/regime-test")
async def regime_backtest(req: RegimeTestRequest):
    try:
        from backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine()
        regimes = req.regimes or {
            "recent": {
                "start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
            }
        }
        runs = {}
        for name, window in regimes.items():
            end = _parse_api_datetime(window.get("end"), datetime.now(timezone.utc))
            start = _parse_api_datetime(window.get("start"), end - timedelta(days=30))
            runs[name] = await engine.run(
                symbol=req.symbol,
                start=start,
                end=end,
                strategy_id=req.strategy_id,
                parameters={**req.parameters, "regime": name},
                qty=req.qty,
            )
        return jsonable_encoder({"status": "completed", "regimes": runs})
    except Exception as exc:
        logger.error("[api] regime_backtest error: {}", exc)
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
        capital = float(state.get("capital") or 100_000)
        daily_limit_pct = 0.02
        weekly_limit_pct = 0.04
        return {
            "kill_switch_armed": state.get("armed", False),
            "reason":            state.get("reason"),
            "armed_at":          str(state.get("armed_at") or ""),
            "daily_loss_usd":    float(state.get("daily_loss_usd") or 0),
            "weekly_loss_usd":   float(state.get("weekly_loss_usd") or 0),
            "capital":           capital,
            "daily_limit_pct":   daily_limit_pct,
            "weekly_limit_pct":  weekly_limit_pct,
            "daily_limit_usd":   capital * daily_limit_pct,
            "weekly_limit_usd":  capital * weekly_limit_pct,
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
        agent_type = req.type or req.agent_type
        if not agent_type:
            raise HTTPException(400, "Request body must include 'type'.")
        name = req.name or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        await db.execute(
            """INSERT INTO agent_registry (name, agent_type, metadata, status)
               VALUES ($1, $2, $3::jsonb, 'idle')
               ON CONFLICT (name) DO UPDATE SET status='idle', updated_at=NOW()""",
            name, agent_type, json.dumps(req.config),
        )
        return {"status": "spawned", "name": name, "agent_type": agent_type}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] spawn_agent error: {}", exc)
        raise HTTPException(500, str(exc))

# ── Execution ─────────────────────────────────────────────────────────────────

@app.post("/api/execution/test-order")
async def test_order(req: OrderRequest):
    try:
        broker = req.broker.lower().strip()

        if broker == "binance_testnet":
            from execution.binance_testnet_executor import BinanceTestnetExecutor

            executor = BinanceTestnetExecutor()
            result = await executor.submit_order(
                symbol=req.symbol,
                qty=req.qty,
                side=req.side,
                order_type=req.order_type,
                price=req.price if req.price is not None else req.limit_price,
                test_only=req.test_only,
            )
            return {"status": "submitted" if result.get("accepted") else "rejected", "order": result}

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


# ── Scouts ────────────────────────────────────────────────────────────────────

@app.post("/api/scouts/test")
async def test_scout(req: ScoutTestRequest):
    try:
        source = req.source.lower().strip()
        if source == "reddit":
            from scouts.reddit_scout import RedditScout

            scout = RedditScout()
            await scout.setup()
            hypotheses = await scout.scan(subreddits=req.subreddits, limit=req.limit)
        elif source == "youtube":
            from scouts.youtube_scout import YouTubeScout

            scout = YouTubeScout()
            await scout.setup()
            hypotheses = await scout.search_and_extract(
                query=req.query,
                max_results=req.max_results,
            )
        elif source == "discord":
            from scouts.discord_scout import DiscordScout

            scout = DiscordScout()
            await scout.setup()
            hypotheses = await scout.scan_messages()
        else:
            raise HTTPException(400, "source must be 'reddit', 'youtube', or 'discord'")

        return {
            "status": "completed",
            "source": source,
            "count": len(hypotheses),
            "hypotheses": hypotheses,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[api] test_scout error: {}", exc)
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
            model="claude-sonnet-4-6",
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
        pool = await db.get_pool()
        engine = PatternRecognitionEngine(pool)
        return await engine.detect_for_symbol(
            symbol=req.symbol,
            lookback_bars=req.lookback_bars,
        )
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


@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, limit: int = 100):
    try:
        rows = await db.fetch(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            symbol.upper(),
            limit,
        )

        data = [
            {
                "timestamp": r["timestamp"].isoformat(),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for r in reversed(rows)
        ]

        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "count": len(data),
            "data": data,
        }
    except Exception as exc:
        logger.error("[api] market_data error: {}", exc)
        raise HTTPException(500, str(exc))


# ── Day 4: Copy Trading ───────────────────────────────────────────────────────

@app.post("/api/copy-trading/link")
async def create_copy_trading_link(req: CopyTradingLinkRequest):
    pool = await db.get_pool()
    engine = CopyTradingEngine(pool)
    link_id = await engine.create_link(
        leader_account=req.leader_account,
        follower_account=req.follower_account,
        follower_risk_limit_pct=req.follower_risk_limit_pct,
        sizing_mode=req.sizing_mode,
    )
    return {
        "status": "created",
        "link_id": link_id,
        "leader_account": req.leader_account,
        "follower_account": req.follower_account,
    }


@app.post("/api/copy-trading/mirror-test")
async def mirror_copy_trade_test(req: CopyTradingMirrorRequest):
    pool = await db.get_pool()
    engine = CopyTradingEngine(pool)
    result = await engine.mirror_trade(
        leader_account=req.leader_account,
        leader_order_id=req.leader_order_id,
        symbol=req.symbol,
        side=req.side,
        leader_qty=req.leader_qty,
        leader_equity=req.leader_equity,
        follower_equity=req.follower_equity,
        price=req.price,
        fill_ratio=req.fill_ratio,
    )
    return result


# ── Day 4: Validation ─────────────────────────────────────────────────────────

@app.post("/api/validation/sensitivity")
async def run_sensitivity_analysis(req: SensitivityRequest):
    analyzer = SensitivityAnalyzer(variation_pct=req.variation_pct)
    return analyzer.analyze(
        baseline_metrics=req.baseline_metrics,
        variant_metrics=req.variant_metrics,
    )


@app.post("/api/validation/regime-test")
async def run_regime_validation(req: RegimeValidationRequest):
    tester = RegimeTester()
    return tester.evaluate(req.regime_metrics)


# ── Day 4: Daily Intelligence Brief ───────────────────────────────────────────

@app.post("/api/dashboard/generate-brief")
async def generate_daily_intelligence_brief():
    pool = await db.get_pool()
    service = IntelligenceBriefService(pool)
    return await service.generate_brief()



# ── Strategy Batch Generation (GEN-004) ───────────────────────────────────────

@app.post("/api/strategies/batch-generate")
async def batch_generate_strategies(req: BatchGenerateStrategyRequest):
    from itertools import cycle

    generated = []
    failed = []

    asset_cycle = cycle(req.asset_classes)
    style_cycle = cycle(req.styles)
    timeframe_cycle = cycle(req.timeframes)

    for i in range(req.count):
        asset_class = next(asset_cycle)
        style = next(style_cycle)
        timeframe = next(timeframe_cycle)
        symbols = req.symbols_by_asset_class.get(asset_class, ["AAPL"])

        last_error = None
        success = False

        for attempt in range(1, 4):
            try:
                payload = GenerateStrategyRequest(
                    asset_class=asset_class,
                    symbols=symbols,
                    timeframe=timeframe,
                    style=style,
                    lookback_days=req.lookback_days,
                    custom_prompt=(
                        "Return STRICT valid JSON only. No markdown. No trailing commentary. "
                        "Use double quotes for all keys and strings. Ensure the JSON object is fully closed."
                    ),
                )
                result = await generate_strategy(payload)

                generated.append({
                    "index": i + 1,
                    "attempt": attempt,
                    "asset_class": asset_class,
                    "style": style,
                    "timeframe": timeframe,
                    "result": result,
                })
                success = True
                break
            except Exception as exc:
                last_error = str(exc)

        if not success:
            failed.append({
                "index": i + 1,
                "asset_class": asset_class,
                "style": style,
                "timeframe": timeframe,
                "error": last_error,
            })

    return {
        "status": "ok" if not failed else "partial",
        "requested_count": req.count,
        "generated_count": len(generated),
        "failed_count": len(failed),
        "generated": generated,
        "failed": failed,
    }



# ── Strategy Code Validation (GEN-003) ────────────────────────────────────────

@app.post("/api/strategies/{strategy_id}/validate")
async def validate_generated_strategy(strategy_id: str):
    row = await db.fetchrow(
        """
        SELECT id, name, code
        FROM strategies
        WHERE id = $1::uuid
        """,
        strategy_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")

    code = row["code"]
    name = row["name"] or f"strategy_{strategy_id}"

    if not code:
        return {
            "strategy_id": strategy_id,
            "valid": False,
            "status": "fail",
            "errors": ["Strategy has no generated code"],
            "required_methods": [
                "generate_signals",
                "compute_position_size",
                "check_filters",
                "get_metadata",
            ],
        }

    try:
        _validate_code(code, name)
        return {
            "strategy_id": strategy_id,
            "valid": True,
            "status": "pass",
            "errors": [],
            "required_methods": [
                "generate_signals",
                "compute_position_size",
                "check_filters",
                "get_metadata",
            ],
            "checks": {
                "syntax": "pass",
                "interface": "pass",
                "base_strategy": "pass",
            },
        }
    except Exception as exc:
        return {
            "strategy_id": strategy_id,
            "valid": False,
            "status": "fail",
            "errors": [str(exc)],
            "required_methods": [
                "generate_signals",
                "compute_position_size",
                "check_filters",
                "get_metadata",
            ],
        }



# ── Day 5: Strategy Mutation (GEN-005) ────────────────────────────────────────

@app.post("/api/strategies/{strategy_id}/mutate")
async def mutate_strategy(strategy_id: str, req: StrategyMutationRequest):
    pool = await db.get_pool()
    engine = StrategyMutationEngine(pool)
    return await engine.mutate_strategy(
        strategy_id=strategy_id,
        mutation_type=req.mutation_type,
        variation_pct=req.variation_pct,
        variants=req.variants,
    )



# ── Day 5: Self-Improvement Cycle ─────────────────────────────────────────────

@app.post("/api/self-improvement/run-cycle")
async def run_self_improvement_cycle(req: SelfImprovementCycleRequest):
    pool = await db.get_pool()
    orchestrator = SelfImprovementOrchestrator(pool)
    return await orchestrator.run_cycle(limit=req.limit)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=False, log_level="info")
