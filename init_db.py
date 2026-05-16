import asyncio
from database import connection as db

async def init_db():
    await db.init_pool()
    print("Dropping old orders/positions/backtests to remove hypertables...")
    await db.execute("DROP TABLE IF EXISTS orders CASCADE;")
    await db.execute("DROP TABLE IF EXISTS positions CASCADE;")
    await db.execute("DROP TABLE IF EXISTS backtests CASCADE;")
    
    with open("database/schema.sql", "r") as f:
        sql = f.read()
    await db.execute(sql)
    print("Schema applied successfully.")
    await db.close_pool()

if __name__ == "__main__":
    asyncio.run(init_db())
