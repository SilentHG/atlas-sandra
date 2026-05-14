"""
data_ingestion package
======================
Market data connectors and stream managers for ATLAS.

Modules
-------
polygon_collector  – Polygon.io WebSocket → 1-min OHLCV for US equities
binance_collector  – Binance WebSocket    → 1-min klines for crypto pairs
alpaca_connector   – Alpaca paper trading REST → order management
ingestor_agent     – BaseAgent wrapper (REST polling, backwards-compat)
"""

from data_ingestion.alpaca_connector import AlpacaConnector
from data_ingestion.binance_collector import run_binance_collector
from data_ingestion.polygon_collector import run_polygon_collector

__all__ = [
    "AlpacaConnector",
    "run_binance_collector",
    "run_polygon_collector",
]
