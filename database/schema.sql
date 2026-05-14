-- ============================================================
-- ATLAS Trading System — TimescaleDB Schema
-- ============================================================
-- Prerequisites:
--   CREATE EXTENSION IF NOT EXISTS timescaledb;  (done below)
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
--
-- Apply:
--   psql -h localhost -U atlas_user -d atlas -f database/schema.sql
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb  CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm     CASCADE;   -- fast text search

-- ============================================================
-- 1. MARKET DATA
--    Core OHLCV time-series — converted to a TimescaleDB
--    hypertable partitioned by timestamp.
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    -- identity
    symbol          TEXT             NOT NULL,
    timestamp       TIMESTAMPTZ      NOT NULL,
    -- OHLCV
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- optional extras
    vwap            DOUBLE PRECISION,
    num_trades      INTEGER,
    exchange        TEXT             NOT NULL DEFAULT 'unknown',
    source          TEXT             NOT NULL DEFAULT 'polygon',
    -- audit
    ingested_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable(
    'market_data', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Continuous aggregate — 1-minute OHLCV candles
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    symbol,
    exchange,
    FIRST(open,  timestamp)            AS open,
    MAX(high)                          AS high,
    MIN(low)                           AS low,
    LAST(close,  timestamp)            AS close,
    SUM(volume)                        AS volume
FROM market_data
GROUP BY bucket, symbol, exchange
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_md_symbol_ts
    ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_md_exchange_ts
    ON market_data (exchange, timestamp DESC);

-- ============================================================
-- 2. FEATURE STORE
--    Key-value store for computed technical / ML features,
--    partitioned by timestamp.
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_store (
    symbol          TEXT             NOT NULL,
    timestamp       TIMESTAMPTZ      NOT NULL,
    feature_name    TEXT             NOT NULL,
    feature_value   DOUBLE PRECISION,
    -- optional structured metadata (e.g. {"window":14,"source":"RSI"})
    feature_meta    JSONB,
    version         INTEGER          NOT NULL DEFAULT 1,
    computed_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol, timestamp, feature_name, version)
);

SELECT create_hypertable(
    'feature_store', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fs_symbol_feature_ts
    ON feature_store (symbol, feature_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fs_meta
    ON feature_store USING GIN (feature_meta);

-- ============================================================
-- 3. STRATEGIES
--    Strategy registry: stores code, parameters, and lifecycle
--    status for every trading strategy in the system.
-- ============================================================
CREATE TABLE IF NOT EXISTS strategies (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT             NOT NULL UNIQUE,
    -- Python source code of the strategy (optional, for dynamic loading)
    code            TEXT,
    -- JSON dict of tunable parameters
    parameters      JSONB            NOT NULL DEFAULT '{}',
    -- lifecycle state: draft | active | paused | archived
    status          TEXT             NOT NULL DEFAULT 'draft'
                                     CHECK (status IN ('draft','active','paused','archived')),
    -- meta
    strategy_type   TEXT             NOT NULL DEFAULT 'custom',
    description     TEXT,
    symbols         TEXT[]           NOT NULL DEFAULT '{}',
    timeframe       TEXT             NOT NULL DEFAULT '1m',
    is_paper        BOOLEAN          NOT NULL DEFAULT TRUE,
    max_position_size DOUBLE PRECISION NOT NULL DEFAULT 0,
    risk_per_trade  DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    -- audit
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strategies_status
    ON strategies (status);
CREATE INDEX IF NOT EXISTS idx_strategies_type
    ON strategies (strategy_type);

-- ============================================================
-- 4. BACKTESTS
--    Results of each strategy backtest run.  Links to strategies
--    via strategy_id (soft reference so historic rows survive
--    strategy deletion).
-- ============================================================
CREATE TABLE IF NOT EXISTS backtests (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID             REFERENCES strategies(id) ON DELETE SET NULL,

    -- ── Core performance metrics ──────────────────────────
    sharpe          DOUBLE PRECISION,
    sortino         DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,   -- stored as a fraction, e.g. -0.12 = -12 %
    win_rate        DOUBLE PRECISION,   -- fraction 0–1

    -- ── Extended metrics (populated when available) ───────
    annualized_return DOUBLE PRECISION,
    total_return    DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    losing_trades   INTEGER,
    avg_win         DOUBLE PRECISION,
    avg_loss        DOUBLE PRECISION,
    initial_capital DOUBLE PRECISION   NOT NULL DEFAULT 10000,
    final_capital   DOUBLE PRECISION,

    -- ── Run metadata ──────────────────────────────────────
    start_date      TIMESTAMPTZ,
    end_date        TIMESTAMPTZ,
    symbols         TEXT[]             NOT NULL DEFAULT '{}',
    parameters      JSONB              NOT NULL DEFAULT '{}',
    -- last 200 equity-curve snapshots: [{time, equity}, ...]
    equity_curve    JSONB,
    -- pending | running | completed | failed
    run_status      TEXT               NOT NULL DEFAULT 'pending'
                                       CHECK (run_status IN ('pending','running','completed','failed')),
    error_message   TEXT,

    -- ── Audit ─────────────────────────────────────────────
    created_at      TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_backtests_strategy
    ON backtests (strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtests_status
    ON backtests (run_status, created_at DESC);

-- ============================================================
-- 5. ORDERS
--    Every order ever submitted, paper or live.
--    Hypertable on filled_at so recent fills are always fast.
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID             REFERENCES strategies(id) ON DELETE SET NULL,

    -- ── Order identity ────────────────────────────────────
    symbol          TEXT             NOT NULL,
    -- buy | sell
    side            TEXT             NOT NULL
                                     CHECK (side IN ('buy','sell')),
    -- market | limit | stop | stop_limit
    order_type      TEXT             NOT NULL DEFAULT 'market'
                                     CHECK (order_type IN ('market','limit','stop','stop_limit')),

    -- ── Sizing ────────────────────────────────────────────
    quantity        DOUBLE PRECISION NOT NULL,
    filled_qty      DOUBLE PRECISION NOT NULL DEFAULT 0,
    limit_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    avg_fill_price  DOUBLE PRECISION,

    -- ── Status ────────────────────────────────────────────
    -- pending | submitted | partial | filled | cancelled | rejected | expired
    status          TEXT             NOT NULL DEFAULT 'pending'
                                     CHECK (status IN
                                        ('pending','submitted','partial',
                                         'filled','cancelled','rejected','expired')),
    -- ── Exchange linkage ──────────────────────────────────
    exchange        TEXT             NOT NULL DEFAULT 'alpaca',
    client_order_id TEXT             UNIQUE,
    exchange_order_id TEXT,
    is_paper        BOOLEAN          NOT NULL DEFAULT TRUE,
    time_in_force   TEXT             NOT NULL DEFAULT 'GTC',
    commission      DOUBLE PRECISION NOT NULL DEFAULT 0,
    raw_response    JSONB,

    -- ── Timestamps ────────────────────────────────────────
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,      -- hypertable partition key
    cancelled_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- Hypertable on filled_at (NULLable) — use created_at as the partition key
-- so all rows are valid even before filling.
SELECT create_hypertable(
    'orders', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_created
    ON orders (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_strategy
    ON orders (strategy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_exchange_id
    ON orders (exchange_order_id)
    WHERE exchange_order_id IS NOT NULL;

-- ============================================================
-- 6. POSITIONS
--    Current and historical positions.  Updated in real-time
--    as fills arrive and marks change.
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID             REFERENCES strategies(id) ON DELETE SET NULL,

    -- ── Instrument ────────────────────────────────────────
    symbol          TEXT             NOT NULL,
    -- long | short
    side            TEXT             NOT NULL DEFAULT 'long'
                                     CHECK (side IN ('long','short')),

    -- ── Sizing & pricing ──────────────────────────────────
    quantity        DOUBLE PRECISION NOT NULL DEFAULT 0,
    entry_price     DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION,

    -- ── P&L (kept up-to-date by a background agent) ───────
    pnl             DOUBLE PRECISION NOT NULL DEFAULT 0,   -- unrealized + realized
    unrealized_pnl  DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl    DOUBLE PRECISION NOT NULL DEFAULT 0,
    commission_paid DOUBLE PRECISION NOT NULL DEFAULT 0,

    -- ── Risk anchors ──────────────────────────────────────
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    trailing_stop   DOUBLE PRECISION,

    -- ── Status ────────────────────────────────────────────
    -- open | partial | closed
    status          TEXT             NOT NULL DEFAULT 'open'
                                     CHECK (status IN ('open','partial','closed')),
    is_paper        BOOLEAN          NOT NULL DEFAULT TRUE,

    -- ── Linked orders ─────────────────────────────────────
    entry_order_id  UUID             REFERENCES orders(id),
    exit_order_id   UUID             REFERENCES orders(id),

    -- ── Timestamps ────────────────────────────────────────
    opened_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    metadata        JSONB,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol_status
    ON positions (symbol, status, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_strategy_status
    ON positions (strategy_id, status);

-- ============================================================
-- 7. AGENT REGISTRY
--    Every autonomous agent registers here.  The orchestrator
--    uses last_heartbeat to detect crashed agents.
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_registry (
    id              UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT             NOT NULL UNIQUE,

    -- ── Classification ────────────────────────────────────
    -- data | signal | execution | risk | orchestrator | dashboard
    agent_type      TEXT             NOT NULL DEFAULT 'custom',
    version         TEXT             NOT NULL DEFAULT '1.0.0',
    description     TEXT,
    capabilities    TEXT[]           NOT NULL DEFAULT '{}',

    -- ── State ─────────────────────────────────────────────
    -- idle | running | paused | error | stopped
    status          TEXT             NOT NULL DEFAULT 'idle'
                                     CHECK (status IN
                                        ('idle','running','paused','error','stopped')),
    last_heartbeat  TIMESTAMPTZ,
    heartbeat_interval_s INTEGER     NOT NULL DEFAULT 30,

    -- ── Error tracking ────────────────────────────────────
    error_count     INTEGER          NOT NULL DEFAULT 0,
    last_error      TEXT,

    -- ── Runtime metrics ───────────────────────────────────
    messages_processed BIGINT        NOT NULL DEFAULT 0,
    uptime_seconds  BIGINT           NOT NULL DEFAULT 0,

    -- ── Arbitrary config / state blob ─────────────────────
    metadata        JSONB            NOT NULL DEFAULT '{}',

    -- ── Audit ─────────────────────────────────────────────
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_registry_status
    ON agent_registry (status);
CREATE INDEX IF NOT EXISTS idx_agent_registry_type
    ON agent_registry (agent_type, status);

-- ============================================================
-- 8. AGENT LOGS  (bonus — required by BaseAgent)
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_logs (
    time            TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    agent_name      TEXT             NOT NULL,
    level           TEXT             NOT NULL DEFAULT 'INFO',
    message         TEXT             NOT NULL,
    metadata        JSONB
);

SELECT create_hypertable(
    'agent_logs', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_agent_logs_name_time
    ON agent_logs (agent_name, time DESC);

-- ============================================================
-- Utility: auto-update updated_at column
-- ============================================================
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY['strategies','orders','positions','agent_registry']
    LOOP
        EXECUTE format($f$
            DROP TRIGGER IF EXISTS trg_%1$s_updated_at ON %1$s;
            CREATE TRIGGER trg_%1$s_updated_at
                BEFORE UPDATE ON %1$s
                FOR EACH ROW EXECUTE FUNCTION _set_updated_at();
        $f$, tbl);
    END LOOP;
END;
$$;

-- ============================================================
-- Seed: default agents
-- ============================================================
INSERT INTO agent_registry (name, agent_type, description, capabilities, metadata)
VALUES
    ('data_ingestor',    'data',        'Ingests OHLCV bars from Polygon & Binance WebSocket',  ARRAY['polygon','binance','websocket','rest'], '{"tick_seconds":60}'),
    ('feature_engineer', 'signal',      'Computes EMA, RSI, MACD, BB, ATR, OBV, VWAP',         ARRAY['ta','pandas_ta','ml_features'],         '{}'),
    ('strategy_runner',  'signal',      'Evaluates registered strategies on live feature data', ARRAY['trend','mean_reversion','momentum'],     '{}'),
    ('risk_manager',     'risk',        'Position sizing, daily-loss cap, drawdown CB',         ARRAY['position_sizing','drawdown','var'],      '{"max_drawdown_pct":0.10,"max_daily_loss":500}'),
    ('execution_agent',  'execution',   'Routes and monitors orders on Alpaca / Binance',       ARRAY['alpaca','binance','paper'],              '{"paper":true}'),
    ('orchestrator',     'orchestrator','Coordinates all agents and enforces health checks',     ARRAY['scheduling','health_check','restart'],   '{"heartbeat_interval_s":30}')
ON CONFLICT (name) DO UPDATE
    SET description = EXCLUDED.description,
        capabilities = EXCLUDED.capabilities,
        metadata     = EXCLUDED.metadata,
        updated_at   = NOW();

-- ============================================================
-- Done
-- ============================================================
DO $$ BEGIN
    RAISE NOTICE 'ATLAS schema applied successfully at %', NOW();
END $$;
