"""
strategy_engine package
========================
- base_strategy   : BaseStrategy, TradeSignal, Signal
- ideator         : Claude-powered strategy spec generator
- strategy_coder  : Claude-powered strategy code generator
"""

from strategy_engine.base_strategy import BaseStrategy, Signal, TradeSignal

__all__ = ["BaseStrategy", "Signal", "TradeSignal"]
