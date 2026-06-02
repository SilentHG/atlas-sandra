import sys, asyncio, uuid, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database import connection as db

CODE = '''
from strategy_engine.base_strategy import BaseStrategy, Signal
import pandas as pd

class LongOnlyCrossoverTemplate(BaseStrategy):
    def __init__(self, name="long_only_template", symbols=["AAPL"], parameters=None):
        super().__init__(name=name, symbols=symbols, parameters=parameters or {})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self.parameters
        fast = int(p.get("fast", 10))
        slow = int(p.get("slow", 30))
        rsi_min = float(p.get("rsi_min", 45))
        rsi_max = float(p.get("rsi_max", 75))

        close = data["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        sig = pd.Series(Signal.HOLD, index=data.index)
        sig[cross_up & (rsi > rsi_min) & (rsi < rsi_max)] = Signal.BUY
        sig[cross_down] = Signal.CLOSE
        return sig
'''

async def main():
    await db.init_pool()
    count = 0
    for symbol in ["AAPL", "NVDA", "MSFT"]:
        for fast in [5, 8, 10, 12, 15, 20]:
            for slow in [30, 40, 50, 75, 100]:
                if fast >= slow:
                    continue
                for rsi_min in [40, 45, 50]:
                    name = f"{symbol.lower()}_longonly_cross_{fast}_{slow}_{rsi_min}"
                    params = {"fast": fast, "slow": slow, "rsi_min": rsi_min, "rsi_max": 80}
                    await db.execute("""
                        INSERT INTO strategies
                        (id,name,code,parameters,status,strategy_type,description,symbols,timeframe,is_paper,max_position_size,risk_per_trade,created_at,updated_at)
                        VALUES ($1,$2,$3,$4::jsonb,'active','live_optimizer','Long-only crossover optimizer template',$5,'1h',true,0.1,0.01,NOW(),NOW())
                        ON CONFLICT (name) DO UPDATE SET code=EXCLUDED.code, parameters=EXCLUDED.parameters, status='active', updated_at=NOW()
                    """, uuid.uuid4(), name, CODE, json.dumps(params), [symbol])
                    count += 1
    await db.close_pool()
    print({"seeded_longonly": count})

asyncio.run(main())
