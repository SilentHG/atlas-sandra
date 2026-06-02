import sys, asyncio, uuid, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database import connection as db

CODE = '''
from strategy_engine.base_strategy import BaseStrategy, Signal
import pandas as pd

class ContrarianTemplateStrategy(BaseStrategy):
    def __init__(self, name="contrarian_template", symbols=["AAPL"], parameters=None):
        super().__init__(name=name, symbols=symbols, parameters=parameters or {})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self.parameters
        fast = int(p.get("fast", 9))
        slow = int(p.get("slow", 21))
        rsi_short = float(p.get("rsi_short", 55))
        rsi_close = float(p.get("rsi_close", 48))

        close = data["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

        sig = pd.Series(Signal.HOLD, index=data.index)

        # Contrarian/short variant: fade overheated momentum
        sig[(ema_fast > ema_slow) & (rsi > rsi_short)] = Signal.SELL
        sig[(ema_fast < ema_slow) | (rsi < rsi_close)] = Signal.CLOSE
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
                for rsi_short in [52, 55, 58, 62]:
                    name = f"{symbol.lower()}_contrarian_ema_rsi_{fast}_{slow}_{rsi_short}"
                    params = {
                        "fast": fast,
                        "slow": slow,
                        "rsi_short": rsi_short,
                        "rsi_close": 48
                    }
                    await db.execute("""
                        INSERT INTO strategies
                        (id,name,code,parameters,status,strategy_type,description,symbols,timeframe,is_paper,max_position_size,risk_per_trade,created_at,updated_at)
                        VALUES ($1,$2,$3,$4::jsonb,'active','optimizer_template','Live contrarian optimizer template',$5,'1h',true,0.1,0.01,NOW(),NOW())
                        ON CONFLICT (name) DO UPDATE SET
                        code=EXCLUDED.code,
                        parameters=EXCLUDED.parameters,
                        status='active',
                        updated_at=NOW()
                    """, uuid.uuid4(), name, CODE, json.dumps(params), [symbol])
                    count += 1

    await db.close_pool()
    print({"seeded_contrarian_templates": count})

asyncio.run(main())
