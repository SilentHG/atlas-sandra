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
      rationale      : str   – plain English explanation of WHY this strategy works
      asset_class    : str   – "equity" or "crypto" or "multi_asset"
      symbols        : list  – list of 2-5 US equity ticker symbols (e.g. ["AAPL","MSFT"])
      timeframe      : str   – bar size e.g. "1m", "5m", "1h"
      entry_rules    : dict  – mapping of rule_name → human-readable condition (MINIMUM 2 rules)
      exit_rules     : dict  – mapping of rule_name → human-readable condition (MINIMUM 2 rules)
      position_sizing: dict  – keys: method (str), parameters (dict with risk_pct, max_pct)
      risk_parameters: dict  – keys: stop_loss (dict), take_profit (dict), max_drawdown (float 0-1)
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
    # 1 — Momentum trend-following
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
    # 2 — Mean-reversion Bollinger
    textwrap.dedent("""\
        Design a mean-reversion strategy for US equities using Bollinger Bands.
        Requirements:
        - Enter long when %B drops below 0.1 AND RSI-14 < 35
        - Confirm with Williams %R < -85
        - Exit when %B crosses above 0.5 OR RSI-14 > 60
        - Stop-loss: 2% hard stop below entry
        - Only trade when CCI-20 shows |CCI| < 100
        - Position size: Kelly criterion max 5% per position
        - Choose symbols from: AAPL, JPM, V, JNJ, PG, KO
    """),
    # 3 — Breakout
    textwrap.dedent("""\
        Design a breakout strategy that enters when price closes above the 20-day high
        with volume > 2× its 20-day average.
        Requirements:
        - Entry: close > highest(close, 20) AND rel_volume_20 > 2.0
        - Confirm with ADX-14 > 25 (trending market)
        - Exit: trailing stop at 2× ATR-14 below the highest close since entry
        - Stop-loss: 1× ATR-14 below breakout level
        - Regime: bullish only (SMA-50 > SMA-200)
        - Position size: risk 1.5% of capital per trade
        - Choose 3–5 symbols from: AAPL, MSFT, NVDA, AMZN, META
    """),
    # 4 — RSI divergence
    textwrap.dedent("""\
        Design an RSI divergence strategy for US equities.
        Requirements:
        - Enter long when price makes a lower low but RSI-14 makes a higher low (bullish divergence)
        - Confirm: stochastic %K crosses above %D from below 20
        - Exit: RSI > 65 or price hits take-profit at 2× risk
        - Stop-loss: below the recent swing low, max 1.5%
        - Regime: any regime, but skip if ADX > 50 (extreme trend)
        - Position size: 1% risk per trade
        - Choose symbols from: TSLA, NVDA, AMD, AMZN, GOOGL
    """),
    # 5 — VWAP reversion
    textwrap.dedent("""\
        Design a VWAP reversion strategy for intraday trading of US equities.
        Requirements:
        - Enter long when price dips > 1% below VWAP AND RSI-7 < 30
        - Enter short when price rises > 1% above VWAP AND RSI-7 > 70
        - Exit: price returns to VWAP or 0.5× ATR profit target
        - Stop-loss: 0.5% beyond entry
        - Regime: ranging market (ADX < 20)
        - Position size: 0.5% risk per trade (tight stops)
        - Choose 3 highly liquid symbols: AAPL, MSFT, AMZN
    """),
    # 6 — Golden cross momentum
    textwrap.dedent("""\
        Design a golden cross momentum strategy.
        Requirements:
        - Enter long when SMA-50 crosses above SMA-200 AND MACD line > signal line
        - Confirm: volume on crossover day > 1.5× average
        - Exit: SMA-50 crosses below SMA-200 OR MACD histogram turns negative for 3 bars
        - Stop-loss: 3× ATR-14 (wide stop for swing trades)
        - Regime: any
        - Position size: 2% risk per trade, max 10% per position
        - Symbols: AAPL, MSFT, GOOGL, V, UNH
    """),
    # 7 — Stochastic oversold bounce
    textwrap.dedent("""\
        Design a stochastic oversold bounce strategy.
        Requirements:
        - Enter long when stoch %K < 20, %D < 20, and %K crosses above %D
        - Confirm: price is above SMA-200 (long-term uptrend intact)
        - Exit: stoch %K > 80 OR RSI-14 > 70
        - Stop-loss: 1.5% below entry
        - Regime: bullish (price > SMA-50)
        - Position size: 1% risk per trade
        - Symbols: JPM, BAC, GS, MS, WFC
    """),
    # 8 — CCI trend-pull
    textwrap.dedent("""\
        Design a CCI trend-pull strategy.
        Requirements:
        - Enter long when CCI-20 crosses above +100 from below (trend beginning)
        - Confirm: EMA-9 > EMA-21 AND volume above average
        - Exit: CCI drops below +100 OR price < EMA-21
        - Stop-loss: 2× ATR-14
        - Regime: trending (ADX > 20)
        - Position size: 1.5% risk per trade
        - Symbols: NVDA, TSLA, AMD, AVGO, QCOM
    """),
    # 9 — Williams %R momentum
    textwrap.dedent("""\
        Design a Williams %R momentum strategy.
        Requirements:
        - Enter long when Williams %R crosses above -50 from below -80 (leaving oversold)
        - Confirm: MACD histogram positive AND price > SMA-20
        - Exit: Williams %R > -20 (overbought) OR trailing stop hit
        - Stop-loss: 1× ATR-14
        - Regime: bullish (price > SMA-100)
        - Position size: 1% risk
        - Symbols: AAPL, MSFT, AMZN, GOOGL, META
    """),
    # 10 — OBV divergence
    textwrap.dedent("""\
        Design an OBV (On Balance Volume) divergence strategy.
        Requirements:
        - Enter long when price makes a lower low but OBV makes a higher low
        - Confirm: RSI-14 < 40 (room to run upward)
        - Exit: OBV diverges bearishly OR RSI > 70
        - Stop-loss: 2% below entry
        - Regime: any, but prefer ranging (ADX < 25)
        - Position size: 1% risk, max 5% per position
        - Symbols: PG, KO, PEP, WMT, COST
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




def _extract_json_object(raw: str) -> str:
    """Extract the first complete JSON object from Claude text."""
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines and lines[-1].strip() == "```" else "\n".join(lines[1:])

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Claude response")

    return text[start:end + 1]


# ── LLM call ─────────────────────────────────────────────────────────────────


async def _generate_spec(
    client: anthropic.Anthropic,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
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

    try:
        spec = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as exc:
        # One retry with a stricter repair instruction.
        logger.warning("[ideator] Claude returned invalid JSON; retrying once with strict JSON-only prompt: {}", exc)

        repair_prompt = (
            user_prompt
            + "\n\nReturn ONLY one complete valid JSON object. "
            + "No markdown, no explanation, no trailing comments. "
            + "All strings must be closed and escaped correctly."
        )

        def _repair_call() -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": repair_prompt}],
            )
            return msg.content[0].text.strip()

        raw2 = await loop.run_in_executor(None, _repair_call)

        try:
            spec = json.loads(_extract_json_object(raw2))
        except json.JSONDecodeError as exc2:
            raise ValueError(f"Claude returned invalid JSON after retry: {exc2}\n---\n{raw2}") from exc2

    # Validate required keys
    required = {
        "name", "description", "symbols", "timeframe",
        "entry_rules", "exit_rules", "stop_loss",
        "position_size", "regime_filter", "parameters", "strategy_type",
        "rationale", "asset_class", "risk_parameters",
    }
    missing = required - set(spec.keys())
    if missing:
        # Try to auto-fill optional fields with defaults
        if "rationale" in missing:
            spec["rationale"] = spec.get("description", "Auto-generated strategy")
            missing.discard("rationale")
        if "asset_class" in missing:
            spec["asset_class"] = "equity"
            missing.discard("asset_class")
        if "risk_parameters" in missing:
            spec["risk_parameters"] = {
                "stop_loss": spec.get("stop_loss", {}),
                "take_profit": {"method": "fixed", "pct": 0.04},
                "max_drawdown": 0.10,
            }
            missing.discard("risk_parameters")
        if "position_sizing" in missing:
            spec["position_sizing"] = spec.get("position_size", {})
            missing.discard("position_sizing")
        if missing:
            raise ValueError(f"Generated spec missing keys: {missing}")

    # Enforce minimum 2 entry rules and 2 exit rules
    entry_rules = spec.get("entry_rules", {})
    exit_rules = spec.get("exit_rules", {})
    if isinstance(entry_rules, dict) and len(entry_rules) < 2:
        raise ValueError(f"entry_rules must have at least 2 conditions, got {len(entry_rules)}")
    if isinstance(exit_rules, dict) and len(exit_rules) < 2:
        raise ValueError(f"exit_rules must have at least 2 conditions, got {len(exit_rules)}")

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
        "entry_rules":     spec.get("entry_rules", {}),
        "exit_rules":      spec.get("exit_rules",  {}),
        "stop_loss":       spec.get("stop_loss",   {}),
        "position_size":   spec.get("position_size", {}),
        "position_sizing": spec.get("position_sizing", spec.get("position_size", {})),
        "regime_filter":   spec.get("regime_filter", {}),
        "risk_parameters": spec.get("risk_parameters", {}),
        "rationale":       spec.get("rationale", ""),
        "asset_class":     spec.get("asset_class", "equity"),
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
    n: int = 10,
    model: str = "claude-sonnet-4-6",
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
    # Cycle through prompts if n > len(_USER_PROMPTS)
    prompts = [_USER_PROMPTS[i % len(_USER_PROMPTS)] for i in range(n)]

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
    # db.close_pool() disabled here; caller owns pool lifecycle.
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
    ids = await run_ideator(n=10)
    if ids:
        print(f"\n[OK] {len(ids)} strategies generated and saved:\n" + "\n".join(f"  - {i}" for i in ids))
    else:
        print("\n[FAIL] No strategies were saved -- check logs above.")


if __name__ == "__main__":
    asyncio.run(_main())


# ── Dynamic ideator (Fix 1 and 2) ────────────────────────────────────────────

def _build_dynamic_prompt(asset_class, symbols, timeframe, style, lookback_days):
    symbol_str = ", ".join(symbols) if symbols else "AAPL, MSFT, NVDA"
    style_hints = {
        "momentum":       "EMA crossovers, RSI confirmation, MACD histogram, trend-following entries",
        "mean_reversion": "Bollinger Bands %B, RSI extremes, Williams %R, reversion to mean",
        "breakout":       "20-day high breakout, volume surge > 2x average, ADX > 25 confirmation",
        "trend":          "SMA/EMA alignment, ADX trend strength, regime filtering with SMA-200",
        "scalping":       "VWAP deviation, short timeframe RSI, tight stops, high frequency",
    }
    asset_hints = {
        "us_equities": "US equity market hours 9:30am-4pm ET, use SMA-200 regime filter",
        "crypto":      "24/7 market, higher volatility, use wider stops, BTC correlation filter",
    }
    style_desc = style_hints.get(style, style_hints["momentum"])
    asset_desc = asset_hints.get(asset_class, asset_hints["us_equities"])
    return f"""Design a {style} strategy for {asset_class} assets.

Symbols to trade: {symbol_str}
Timeframe: {timeframe}
Lookback period: {lookback_days} days
Market context: {asset_desc}

Strategy requirements:
- Style: {style} — use {style_desc}
- Entry conditions: define precise indicator thresholds with exact numeric values
- Exit conditions: both take-profit and stop-loss rules required
- Stop-loss: ATR-based preferred, define multiplier explicitly
- Position sizing: risk 1-2% of capital per trade
- Regime filter: only trade in appropriate market conditions
- Use only these available features: sma_5, sma_10, sma_20, sma_50, sma_200,
  ema_9, ema_21, rsi_7, rsi_14, macd, macd_signal, macd_hist, bb_upper,
  bb_lower, bb_position, bb_bandwidth, atr_14, atr_14_pct, vwap,
  stoch_k, stoch_d, williams_r, cci_20, adx_14, obv_signal, roc_20,
  vol_sma_ratio_20, hl_range, hl_range_pct, gap_flag, gap_pct,
  regime_score, regime_trend, regime_high_vol, golden_cross,
  dist_sma50_pct, corr_msft_20, corr_nvda_20, corr_tsla_20, corr_aapl_20

Return a complete JSON strategy specification."""


async def run_ideator_dynamic(
    asset_class="us_equities",
    symbols=None,
    timeframe="1h",
    style="momentum",
    lookback_days=90,
    custom_prompt=None,
    model="claude-sonnet-4-6",
):
    """Generate a strategy dynamically from request parameters."""
    await db.init_pool()
    client = _get_claude_client()
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA"]
    prompt = custom_prompt or _build_dynamic_prompt(asset_class, symbols, timeframe, style, lookback_days)
    logger.info("[ideator] Dynamic generation | asset={} symbols={} timeframe={} style={}", asset_class, symbols, timeframe, style)
    saved_ids = []
    try:
        spec = await _generate_spec(client, prompt, model=model)
        spec["_generation_meta"] = {
            "user_prompt": prompt,
            "asset_class": asset_class,
            "symbols": symbols,
            "timeframe": timeframe,
            "style": style,
            "lookback_days": lookback_days,
            "model": model,
        }
        logger.info("[ideator] Generated: '{}' ({})", spec["name"], spec["strategy_type"])
        sid = await _save_strategy(spec)
        saved_ids.append(sid)
        logger.info("[ideator] Strategy saved: {}", sid)
    except Exception as exc:
        logger.error("[ideator] Dynamic generation failed: {}", exc, exc_info=True)
        raise
        raise
    # db.close_pool() disabled here; caller owns pool lifecycle.
    return saved_ids
