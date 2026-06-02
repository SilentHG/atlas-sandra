"""
ATLAS Factory Worker

Runs strategy generation/coding/backtesting OUTSIDE the FastAPI dashboard process.
This prevents dashboard hangs and meets the buyer requirement for bulk strategy evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timezone, timedelta

from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

from database import connection as db
from strategy_engine.ideator import run_ideator_dynamic
from strategy_engine.strategy_coder import run_strategy_coder
from backtesting.backtest_engine import BacktestEngine


JOBS = [
    {"asset_class": "us_equities", "symbols": ["AAPL"], "timeframe": "1h", "style": "momentum"},
    {"asset_class": "us_equities", "symbols": ["NVDA"], "timeframe": "1h", "style": "breakout"},
    {"asset_class": "us_equities", "symbols": ["MSFT"], "timeframe": "1h", "style": "trend"},
]


async def run_factory(target: int = 10, promote_top: int = 3):
    await db.init_pool()

    generated = 0
    coded = 0
    backtested = 0
    failures = 0

    engine = BacktestEngine()

    try:
        while generated < target:
            for job in JOBS:
                if generated >= target:
                    break

                try:
                    logger.info("[factory_worker] generating {}", job)

                    ids = await run_ideator_dynamic(
                        asset_class=job["asset_class"],
                        symbols=job["symbols"],
                        timeframe=job["timeframe"],
                        style=job["style"],
                        lookback_days=180,
                        custom_prompt=(
                            "Generate a HIGH FREQUENCY systematic trading strategy. "
                            "The strategy MUST create between 20 and 200 trades during a 2 year backtest. "
                            "Use ONLY these indicators: EMA, SMA, RSI, MACD, VWAP, ATR, Bollinger Bands, Volume. "
                            "DO NOT use VIX filters, session filters, market regime filters, macro filters, "
                            "multi-timeframe confirmation, daily trend filters, or excessive confirmations. "
                            "Maximum 3 entry conditions. Maximum 2 exit conditions. "
                            "Prioritize trade frequency over perfection. "
                            "Generate realistic long-only momentum, mean-reversion, or breakout systems "
                            "that can produce dozens of trades. "
                            "Use only symbols: " + ", ".join(job["symbols"])
                        ),
                    )

                    generated += len(ids)

                    for sid in ids:
                        try:
                            coded_ids = await run_strategy_coder(strategy_id=sid)
                            coded += len(coded_ids)

                            for coded_sid in coded_ids:
                                symbol = job["symbols"][0]
                                logger.info("[factory_worker] backtesting {} on {}", coded_sid, symbol)

                                # Suppress noisy generated-strategy logs during bulk backtests.
                                import contextlib, os

                                with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                                    await engine.run(
                                        symbol=symbol,
                                        strategy_id=coded_sid,
                                        start=datetime.now(timezone.utc) - timedelta(days=90),
                                        end=datetime.now(timezone.utc),
                                        enforce_min_history=False,
                                    )

                                backtested += 1

                        except Exception as exc:
                            failures += 1
                            logger.exception("[factory_worker] coding/backtest failed for {}: {}", sid, exc)

                except Exception as exc:
                    failures += 1
                    logger.exception("[factory_worker] job failed {}: {}", job, exc)

        logger.info(
            "[factory_worker] completed generated={} coded={} backtested={} failures={}",
            generated, coded, backtested, failures
        )

        return {
            "generated": generated,
            "coded": coded,
            "backtested": backtested,
            "failures": failures,
        }

    finally:
        await db.close_pool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--promote-top", type=int, default=3)
    args = parser.parse_args()

    result = asyncio.run(run_factory(target=args.target, promote_top=args.promote_top))
    print(result)


if __name__ == "__main__":
    main()
