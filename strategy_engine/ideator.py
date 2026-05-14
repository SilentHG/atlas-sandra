"""
ATLAS Strategy Ideator  ─ Day 2
================================
Uses the Anthropic Claude API to generate complete, structured trading
strategy specifications for US equities, then persists them to the
``strategies`` table in TimescaleDB.

Each generated specification includes:
  • entry_rules        – precise signal conditions (indicator thresholds, etc.)
  • exit_rules         – take-profit, trailing-stop, time-based exits
  • stop_loss          – hard stop methodology and percentage
  • position_size      – sizing model (fixed-fractional, Kelly, etc.)
  • regime_filter      – market-regime gate that must pass before entry

Usage
-----
    # Generate 2 strategies (default) and save them:
    python -m strategy_engine.ideator

    # Programmatic:
    from strategy_engine.ideator import run_ideator
    asyncio.run(run_ideator(n=2))

Credentials loaded from config/keys.env via ATLASSettings.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic
from loguru import logger

from config.settings import settings
from database import connection as db

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a quantitative trading strategy designer specialising in US equities.
    You design clear, rule-based, algorithmic trading strategies that can be
    implemented in Python using standard technical indicators.

    For every strategy you output a single, valid JSON object with exactly these keys:
      name           : str   – short unique strategy name (snake_case)
      description    : str   – 2-3 sentence plain-English description
      symbols        : list  – list of 2-5 US equity ticker symbols (e.g. ["AAPL","MSFT"])
      timeframe      : str   – bar size e.g. "1m", "5m", "1h"
      entry_rules    : dict  – mapping of rule_name → human-readable condition
      exit_rules     : dict  – mapping of rule_name → human-readable condition
      stop_loss      : dict  – keys: method (str), pct (float 0-1), atr_multiple (float)
      position_size  : dict  – keys: method (str), risk_pct (float 0-1), max_pct (float 0-1)
      regime_filter  : dict  – keys: required_trend (str: "bullish"|"bearish"|"any"),
                                      min_adx (float), max_vix (float|null),
                                      description (str)
      parameters     : dict  – tunable numeric parameters (indicator periods, thresholds)
      strategy_type  : str   – one of "trend"|"mean_reversion"|"momentum"|"breakout"

    Output ONLY the raw JSON object. No markdown, no explanation.
""")

_USER_PROMPTS: list[str] = [
    # Strategy 1 — momentum / trend-following
    textwrap.dedent("""\
        Design a momentum trend-following strategy for large-cap US tech equities
        (choose 3–5 symbols from AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META).

        Requirements:
        - Enter long when price crosses above both EMA-9 and EMA-21 with RSI-14 between 50–70
        - Use MACD histogram turning positive as confirmation
        - Exit when RSI-14 > 75 OR price crosses below EMA-21
        - Stop-loss: 1.5× ATR-14 below entry
        - Only trade when the daily regime is bullish (price > SMA-200)
        - Position size: fixed fractional, risk 1% of capital per trade
    """),
    # Strategy 2 — mean-reversion
    textwrap.dedent("""\
        Design a mean-reversion strategy for US equities using Bollinger Bands.

        Requirements:
        - Enter long when %B drops below 0.1 (close near lower band) AND RSI-14 < 35
        - Confirm with Williams %R < -85 (oversold)
        - Exit when %B crosses above 0.5 OR RSI-14 > 60
        - Stop-loss: 2% hard stop below entry
        - Only trade when CCI-20 shows the market is not in a strong trend (|CCI| < 100)
        - Position size: Kelly criterion with max 5% of portfolio per position
        - Choose 3–5 symbols from SP500 liquid names: AAPL, JPM, V, JNJ, PG, KO
    """),
]

# ── Claude client factory ─────────────────────────────────────────────────────


def _get_claude_client() -> anthropic.Anthropic:
    """Return an authenticated Anthropic client."""
    key = settings.anthropic_api_key
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to config/keys.env and restart."
        )
    return anthropic.Anthropic(api_key=key)


# ── LLM call ─────────────────────────────────────────────────────────────────


async def _generate_spec(
    client: anthropic.Anthropic,
    user_prompt: str,
    model: str = "claude-opus-4-5",
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Call Claude synchronously in a thread pool (Anthropic SDK is sync),
    parse the JSON response and return the strategy spec dict.
    """
    def _call() -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text.strip()

    logger.info("[ideator] Calling Claude ({}) …", model)
    loop = asyncio.get_running_loop()
    raw  = await loop.run_in_executor(None, _call)
    logger.debug("[ideator] Raw response ({} chars)", len(raw))

    # Strip markdown code fences if the model wrapped the JSON
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw   = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {exc}\n---\n{raw}") from exc

    # Validate required keys
    required = {
        "name", "description", "symbols", "timeframe",
        "entry_rules", "exit_rules", "stop_loss",
        "position_size", "regime_filter", "parameters", "strategy_type",
    }
    missing = required - set(spec.keys())
    if missing:
        raise ValueError(f"Generated spec missing keys: {missing}")

    return spec


# ── DB persistence ────────────────────────────────────────────────────────────

_UPSERT_SQL = """
    INSERT INTO strategies
        (id, name, description, symbols, timeframe,
         strategy_type, parameters, status,
         risk_per_trade, max_position_size, is_paper,
         created_at, updated_at)
    VALUES
        ($1, $2, $3, $4, $5,
         $6, $7::jsonb, 'draft',
         $8, $9, TRUE,
         NOW(), NOW())
    ON CONFLICT (name) DO UPDATE
        SET description      = EXCLUDED.description,
            symbols          = EXCLUDED.symbols,
            timeframe        = EXCLUDED.timeframe,
            strategy_type    = EXCLUDED.strategy_type,
            parameters       = EXCLUDED.parameters,
            risk_per_trade   = EXCLUDED.risk_per_trade,
            max_position_size= EXCLUDED.max_position_size,
            updated_at       = NOW()
    RETURNING id
"""


async def _save_strategy(spec: dict[str, Any]) -> str:
    """
    Persist the strategy spec to the strategies table.
    The raw spec (entry_rules, exit_rules, stop_loss, regime_filter, etc.)
    is merged into the `parameters` JSONB column for full queryability.
    Returns the strategy UUID.
    """
    sid = str(uuid.uuid4())

    # Pack the full spec into parameters for downstream use
    params = {
        "entry_rules":   spec.get("entry_rules", {}),
        "exit_rules":    spec.get("exit_rules",  {}),
        "stop_loss":     spec.get("stop_loss",   {}),
        "position_size": spec.get("position_size", {}),
        "regime_filter": spec.get("regime_filter", {}),
        **spec.get("parameters", {}),
    }

    risk_pct  = float(
        spec.get("position_size", {}).get("risk_pct", 0.01)
    )
    max_pct   = float(
        spec.get("position_size", {}).get("max_pct", 0.05)
    )

    rows = await db.fetch(
        _UPSERT_SQL,
        sid,
        spec["name"],
        spec.get("description", ""),
        spec.get("symbols", []),
        spec.get("timeframe", "1m"),
        spec.get("strategy_type", "custom"),
        json.dumps(params),
        risk_pct,
        max_pct,
    )

    returned_id = str(rows[0]["id"]) if rows else sid
    logger.info(
        "[ideator] Strategy '{}' saved → id={}", spec["name"], returned_id
    )
    return returned_id


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def run_ideator(
    n: int = 2,
    model: str = "claude-opus-4-5",
) -> list[str]:
    """
    Generate `n` strategy specs via Claude and save them to the DB.

    Parameters
    ----------
    n : int
        Number of strategies to generate (uses built-in prompts cyclically).
    model : str
        Anthropic model identifier.

    Returns
    -------
    list[str]
        UUIDs of saved strategy rows.
    """
    await db.init_pool()
    client = _get_claude_client()
    prompts = _USER_PROMPTS[:n]   # use first n built-in prompts

    saved_ids: list[str] = []
    for idx, prompt in enumerate(prompts, start=1):
        logger.info("[ideator] Generating strategy {}/{} …", idx, len(prompts))
        try:
            spec = await _generate_spec(client, prompt, model=model)
            logger.info(
                "[ideator] Generated: '{}' ({})", spec["name"], spec["strategy_type"]
            )
            sid = await _save_strategy(spec)
            saved_ids.append(sid)
        except Exception as exc:
            logger.error("[ideator] Failed for prompt {}: {}", idx, exc, exc_info=True)

    logger.info(
        "[ideator] Done. {}/{} strategies saved: {}",
        len(saved_ids), len(prompts), saved_ids,
    )
    await db.close_pool()
    return saved_ids


# ── CLI ────────────────────────────────────────────────────────────────────────


async def _main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
    )
    ids = await run_ideator(n=2)
    if ids:
        print(f"\n✅  {len(ids)} strategies generated and saved:\n" + "\n".join(f"  • {i}" for i in ids))
    else:
        print("\n❌  No strategies were saved — check logs above.")


if __name__ == "__main__":
    asyncio.run(_main())
