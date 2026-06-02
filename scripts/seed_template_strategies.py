import asyncio, uuid, json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import connection as db

TEMPLATE = '''
from strategy_engine.base_strategy import BaseStrategy, Signal
import pandas as pd

class TemplateStrategy(BaseStrategy):
    def __init__(self, name="template_strategy", symbols=["AAPL"], parameters=None):
        super().__init__(name=name, symbols=symbols, parameters=parameters or {})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self.parameters
        fast = int(p.get("fast", 9))
        slow = int(p.get("slow", 21))
        rsi_buy = float(p.get("rsi_buy", 55))
        rsi_sell = float(p.get("rsi_sell", 45))

        close = data["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

        sig = pd.Series(Signal.HOLD, index=data.index)
        sig[(ema_fast > ema_slow) & (rsi > rsi_buy)] = Signal.BUY
        sig[(ema_fast < ema_slow) | (rsi < rsi_sell)] = Signal.CLOSE
        return sig
'''

async def main():
    await db.init_pool()
    symbols = ["AAPL", "NVDA", "MSFT"]
    count = 0

    for symbol in symbols:
        for fast in [5, 8, 9, 12, 15]:
            for slow in [20, 26, 34, 50]:
                if fast >= slow:
                    continue
                name = f"{symbol.lower()}_template_ema_rsi_{fast}_{slow}"
                params = {"fast": fast, "slow": slow, "rsi_buy": 52, "rsi_sell": 48}
                await db.execute("""
                    INSERT INTO strategies
                    (id,name,code,parameters,status,strategy_type,description,symbols,timeframe,is_paper,max_position_size,risk_per_trade,created_at,updated_at)
                    VALUES ($1,$2,$3,$4::jsonb,'active','template','Fast template EMA/RSI strategy',$5,'1h',true,0.1,0.01,NOW(),NOW())
                    ON CONFLICT (name) DO UPDATE SET
                    code=EXCLUDED.code,
                    parameters=EXCLUDED.parameters,
                    status='active',
                    updated_at=NOW()
                """, uuid.uuid4(), name, TEMPLATE, json.dumps(params), [symbol])
                count += 1

    await db.close_pool()
    print({"seeded_templates": count})

asyncio.run(main())
