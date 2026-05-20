"""
ATLAS Strategy Coder  ─ Day 2
==============================
Reads strategy specifications from the ``strategies`` table (status=draft,
code IS NULL), generates executable Python code via Claude, and writes
the code back to the ``code`` column (status → 'active').

The generated code conforms to the ``BaseStrategy`` interface:
    class <StrategyName>(BaseStrategy):
        def generate_signal(self, symbol, features) -> TradeSignal: ...

Usage
-----
    python -m strategy_engine.strategy_coder        # code all pending drafts
    python -m strategy_engine.strategy_coder --id <uuid>   # code one strategy

Credentials loaded from config/keys.env via ATLASSettings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from config.settings import settings
from database import connection as db

# ── Code-generation prompt ────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Python quantitative developer.
    You receive a trading strategy specification as JSON and output a single,
    self-contained Python module that implements the strategy.

    Rules:
    1. The class MUST extend BaseStrategy from strategy_engine.base_strategy.
    2. The class MUST implement ALL of these exact methods:
       a. generate_signals(self, data: pd.DataFrame) -> pd.Series
          - Core signal logic. The `data` DataFrame has one row per bar;
            the most recent bar is the LAST row.
            Column names match ATLAS feature names (e.g. rsi_14, macd_line, bb_pct_b).
          - Return a pd.Series with values from Signal enum: BUY, SELL, HOLD, or CLOSE.
       b. compute_position_size(self, signal, portfolio_value: float) -> float
          - Return the number of shares/units to trade based on risk parameters.
          - signal can be a Signal enum or TradeSignal object.
       c. check_filters(self, data: pd.DataFrame) -> bool
          - Return True if regime/market filters pass and trading is allowed.
          - Must check regime_filter conditions from the spec.
       d. get_metadata(self) -> dict
          - Return a dict with: name, version, description, strategy_type, symbols, parameters.
    3. Set stop_loss, take_profit, confidence where appropriate in TradeSignal.
    4. Include a module-level docstring and inline comments.
    5. Do NOT import external libraries beyond: pandas, numpy, uuid, loguru.
    6. Output ONLY the Python source code. No markdown, no explanation.
    7. All numeric thresholds must come from self.parameters so they are tunable.
    8. Guard every indicator lookup with a check that the column exists and is non-NaN.
""")

_USER_TEMPLATE = textwrap.dedent("""\
    Generate a complete Python class for this ATLAS trading strategy.

    Strategy specification (JSON):
    {spec_json}

    The class name must be: {class_name}

    Required imports at the top of the file:
        from __future__ import annotations
        import uuid
        import numpy as np
        import pandas as pd
        from loguru import logger
        from strategy_engine.base_strategy import BaseStrategy, TradeSignal, Signal

    Make sure self.parameters is used for all tunable thresholds.
""")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_class_name(strategy_name: str) -> str:
    """Convert snake_case strategy name to CamelCase class name."""
    return "".join(word.capitalize() for word in re.split(r"[_\-\s]+", strategy_name))


def _get_claude_client() -> anthropic.Anthropic:
    key = settings.anthropic_api_key
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in config/keys.env")
    return anthropic.Anthropic(api_key=key)


# ── LLM call ─────────────────────────────────────────────────────────────────


async def _generate_code(
    client: anthropic.Anthropic,
    strategy_row: dict[str, Any],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8192,
) -> str:
    """Ask Claude to produce executable Python for the strategy spec."""
    name       = strategy_row["name"]
    class_name = _to_class_name(name)
    params     = strategy_row.get("parameters") or {}
    if isinstance(params, str):
        import json
        params = json.loads(params)

    # Build a clean spec dict for the prompt
    spec = {
        "name":          name,
        "description":   strategy_row.get("description", ""),
        "symbols":       strategy_row.get("symbols", []),
        "timeframe":     strategy_row.get("timeframe", "1m"),
        "strategy_type": strategy_row.get("strategy_type", "custom"),
        "parameters":    params,
        # Extract rule sub-dicts from parameters JSONB
        "entry_rules":   params.get("entry_rules",   {}),
        "exit_rules":    params.get("exit_rules",    {}),
        "stop_loss":     params.get("stop_loss",     {}),
        "position_size": params.get("position_size", {}),
        "regime_filter": params.get("regime_filter", {}),
    }

    user_msg = _USER_TEMPLATE.format(
        spec_json=json.dumps(spec, indent=2),
        class_name=class_name,
    )

    def _call() -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return msg.content[0].text.strip()

    logger.info("[strategy_coder] Calling Claude for '{}' …", name)
    loop = asyncio.get_running_loop()
    raw  = await loop.run_in_executor(None, _call)

    # Strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw   = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    return raw


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_code(code: str, strategy_name: str) -> None:
    """Compile-check the generated code and verify class structure."""
    try:
        compiled = compile(code, f"<strategy:{strategy_name}>", "exec")  # noqa: F841
    except SyntaxError as exc:
        raise ValueError(f"Generated code has syntax error: {exc}") from exc

    required_methods = [
        "generate_signals",
        "compute_position_size",
        "check_filters",
        "get_metadata",
    ]
    missing = [m for m in required_methods if m not in code]
    if missing:
        raise ValueError(f"Generated code missing required methods: {missing}")
    if "BaseStrategy" not in code:
        raise ValueError("Generated code does not extend BaseStrategy")

    # Also validate with ast.parse for stricter syntax checking
    import ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"AST parse error: {exc}") from exc

    logger.debug("[strategy_coder] Code validation passed for '{}'", strategy_name)


# ── DB helpers ────────────────────────────────────────────────────────────────

_FETCH_PENDING_SQL = """
    SELECT id, name, description, symbols, timeframe,
           strategy_type, parameters, risk_per_trade, max_position_size
      FROM strategies
     WHERE status = 'draft'
       AND (code IS NULL OR code = '')
     ORDER BY created_at ASC
"""

_FETCH_BY_ID_SQL = """
    SELECT id, name, description, symbols, timeframe,
           strategy_type, parameters, risk_per_trade, max_position_size
      FROM strategies
     WHERE id = $1
"""

_UPDATE_CODE_SQL = """
    UPDATE strategies
       SET code       = $1,
           status     = 'active',
           updated_at = NOW()
     WHERE id = $2
"""


async def _load_pending(strategy_id: str | None = None) -> list[dict[str, Any]]:
    """Load strategies that need code generation."""
    if strategy_id:
        rows = await db.fetch(_FETCH_BY_ID_SQL, strategy_id)
    else:
        rows = await db.fetch(_FETCH_PENDING_SQL)
    return [dict(r) for r in rows]


async def _save_code(strategy_id: str, code: str) -> None:
    await db.execute(_UPDATE_CODE_SQL, code, strategy_id)
    generated_dir = Path("atlas/strategies/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)
    row = await db.fetchrow("SELECT name FROM strategies WHERE id = $1", strategy_id)
    strategy_name = row["name"] if row else strategy_id
    file_name = re.sub(r"[^a-zA-Z0-9_]+", "_", strategy_name).strip("_").lower()
    (generated_dir / f"{file_name}.py").write_text(code, encoding="utf-8")
    logger.info("[strategy_coder] Code saved and status → active for {}", strategy_id)


# ── Main pipeline ─────────────────────────────────────────────────────────────


async def run_strategy_coder(
    strategy_id: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> list[str]:
    """
    Code all pending draft strategies (or a specific one by UUID).

    Returns
    -------
    list[str]
        UUIDs of strategies successfully coded and activated.
    """
    await db.init_pool()
    client  = _get_claude_client()
    rows    = await _load_pending(strategy_id)

    if not rows:
        logger.info("[strategy_coder] No pending strategies found.")
        await db.close_pool()
        return []

    logger.info("[strategy_coder] {} strategies to code", len(rows))
    coded: list[str] = []

    for row in rows:
        sid  = str(row["id"])
        name = row["name"]
        logger.info("[strategy_coder] Processing '{}' ({})", name, sid)
        try:
            code = await _generate_code(client, row, model=model)
            _validate_code(code, name)
            await _save_code(sid, code)
            coded.append(sid)
            logger.info("[strategy_coder] [OK] '{}' activated ({} chars)", name, len(code))
        except Exception as exc:
            logger.error(
                "[strategy_coder] [FAIL] Failed for '{}': {}",
                name, exc, exc_info=True,
            )

    await db.close_pool()
    logger.info(
        "[strategy_coder] Done. {}/{} strategies coded: {}",
        len(coded), len(rows), coded,
    )
    return coded


# ── CLI ────────────────────────────────────────────────────────────────────────


async def _main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS Strategy Coder")
    parser.add_argument("--id", dest="strategy_id", default=None,
                        help="UUID of a specific strategy to code (default: all pending drafts)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model to use")
    args = parser.parse_args()

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

    ids = await run_strategy_coder(strategy_id=args.strategy_id, model=args.model)
    if ids:
        print(f"\n[OK]  {len(ids)} strategies coded and activated:\n" + "\n".join(f"  - {i}" for i in ids))
    else:
        print("\n[FAIL] No strategies were coded -- check logs above.")


if __name__ == "__main__":
    asyncio.run(_main())
