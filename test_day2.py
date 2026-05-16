import asyncio
import uuid
import pandas as pd
from loguru import logger
from database import connection as db
from risk_management.risk_manager import RiskManager
from execution.alpaca_executor import AlpacaExecutor
from strategy_engine.base_strategy import TradeSignal, Signal

async def test_risk():
    logger.info("=== Testing RISK-001 (Max Qty > 1000) ===")
    rm = RiskManager(capital=10_000.0, max_risk_per_trade=0.02)
    # Stop distance is small (0.01), so $200 risk / 0.01 = 20,000 shares
    signal = TradeSignal(
        symbol="AAPL",
        signal=Signal.BUY,
        strategy_id="test-strat-001",
        strategy_name="test_strat",
        confidence=0.9,
        stop_loss=9.99,
        take_profit=10.50
    )
    res = rm.check_signal(signal, current_price=10.00)
    logger.info(f"Risk Check Result: Approved={res.approved}, Reason={res.rejection_reason}")

async def test_execution():
    logger.info("=== Testing Alpaca Execution Skeleton ===")
    await db.init_pool()
    from risk_management.kill_switch import get_kill_switch
    await get_kill_switch().setup()
    
    exec_engine = AlpacaExecutor()
    await exec_engine.setup()
    try:
        acc = await exec_engine.get_account()
        logger.info(f"Alpaca Account Connected: Status={acc.get('status')}, Equity={acc.get('equity')}")
        
        logger.info("Submitting test order for 1 AAPL...")
        order = await exec_engine.submit_order(
            symbol="AAPL",
            qty=1.0,
            side="buy",
            order_type="market"
        )
        logger.info(f"Order submitted! ID: {order.get('id')}")
        
        # Give it a second to persist
        await asyncio.sleep(1)
        
        # Check DB
        client_order_id = order.get("client_order_id")
        row = await db.fetchrow("SELECT * FROM orders WHERE client_order_id=$1", client_order_id)
        if row:
            logger.info(f"Order saved to DB: Status={row['status']}, Symbol={row['symbol']}, Qty={row['quantity']}")
        else:
            logger.error("Order NOT found in DB!")
            
    except Exception as e:
        logger.error(f"Execution test failed: {e}")
    finally:
        await exec_engine.teardown()
        await db.close_pool()

async def main():
    await test_risk()
    print("\n")
    await test_execution()

if __name__ == "__main__":
    asyncio.run(main())
