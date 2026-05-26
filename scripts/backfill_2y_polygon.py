import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from database import connection as db
from data_ingestion.polygon_collector import _rest_backfill_range

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]


def month_windows(start: datetime, end: datetime):
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while cur < end:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)

        win_start = max(cur, start)
        win_end = min(nxt - timedelta(seconds=1), end)

        if win_start < win_end:
            yield win_start, win_end

        cur = nxt


async def main():
    await db.init_pool()

    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=730)

    print(f"Backfilling Polygon 2y data month-by-month: {start} -> {end}")

    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        total = 0

        for win_start, win_end in month_windows(start, end):
            print(f"{symbol}: {win_start.date()} -> {win_end.date()}")
            count = await _rest_backfill_range(
                symbol,
                win_start,
                win_end,
                settings.polygon_api_key,
                force=True,
            )
            total += count
            await asyncio.sleep(0.25)

        print(f"{symbol}: total inserted/updated bars: {total}")

    await db.close_pool()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
