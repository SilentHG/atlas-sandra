import sys, asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database import connection as db
from backtesting.backtest_engine import BacktestEngine

async def main():
    await db.init_pool()
    rows = await db.fetch("""
        SELECT id,name,symbols
        FROM strategies
        WHERE strategy_type='optimizer_template'
        ORDER BY updated_at DESC
        LIMIT 120
    """)

    engine = BacktestEngine()
    tested = 0
    failed = 0

    for row in rows:
        try:
            symbol = row["symbols"][0]
            print(f"Backtesting {row['name']} on {symbol}")
            await engine.run(
                symbol=symbol,
                strategy_id=str(row["id"]),
                start=datetime.now(timezone.utc)-timedelta(days=90),
                end=datetime.now(timezone.utc),
                enforce_min_history=False
            )
            tested += 1
        except Exception as e:
            print("FAILED", row["name"], e)
            failed += 1

    await db.close_pool()
    print({"tested": tested, "failed": failed})

asyncio.run(main())
