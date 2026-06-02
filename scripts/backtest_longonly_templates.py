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
        WHERE strategy_type='live_optimizer'
        ORDER BY updated_at DESC
        LIMIT 90
    """)
    engine = BacktestEngine()
    tested = 0
    for row in rows:
        print("Backtesting", row["name"])
        await engine.run(
            symbol=row["symbols"][0],
            strategy_id=str(row["id"]),
            start=datetime.now(timezone.utc)-timedelta(days=90),
            end=datetime.now(timezone.utc),
            enforce_min_history=False
        )
        tested += 1
    await db.close_pool()
    print({"tested": tested})

asyncio.run(main())
