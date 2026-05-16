"""
ATLAS Base Strategy
====================
Base contract for all trading strategies.
Concrete strategies can override the full vectorized interface:
  - generate_signals(data) -> pd.Series
  - compute_position_size(signal, portfolio_value) -> float
  - check_filters(data) -> bool
  - get_metadata() -> dict
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger


class Signal(str, Enum):
    BUY   = "BUY"
    SELL  = "SELL"
    HOLD  = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class TradeSignal:
    strategy_id:   uuid.UUID
    strategy_name: str
    symbol:        str
    signal:        Signal
    confidence:    float          = 0.0        # 0.0 – 1.0
    entry_price:   float | None   = None
    stop_loss:     float | None   = None
    take_profit:   float | None   = None
    metadata:      dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Base class for all ATLAS strategies.

    ATLAS supports two strategy contracts:
    - the current vectorized ``generate_signals`` method used by generated code
    - the legacy single-symbol ``generate_signal`` method used by earlier tests

    The default implementations bridge those contracts so older strategies keep
    running while new generated strategies can override the richer methods.
    """

    strategy_type: str = "base"

    def __init__(self, name: str, symbols: list[str], parameters: dict[str, Any] | None = None) -> None:
        self.id         = uuid.uuid4()
        self.name       = name
        self.symbols    = symbols
        self.parameters = parameters or {}
        self.is_active  = False
        self.is_paper   = True

    # ── Strategy interface (GEN-001, GEN-002, GEN-003) ───────────────────────

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Given a DataFrame of OHLCV + features (latest row = most recent),
        return a pd.Series of Signal values (BUY, SELL, HOLD, CLOSE).
        """
        if data.empty:
            return pd.Series(dtype=object)

        signals: list[Signal] = []
        for idx in range(len(data)):
            window = data.iloc[: idx + 1]
            symbol = self.symbols[0] if self.symbols else ""
            signals.append(self.generate_signal(symbol, window).signal)
        return pd.Series(signals, index=data.index)

    def compute_position_size(self, signal: Any, portfolio_value: float) -> float:
        """
        Given a signal and current portfolio value, return the number of
        shares/units to trade.
        """
        risk_pct = float(self.parameters.get("risk_pct", self.parameters.get("risk_per_trade", 0.01)))
        entry_price = getattr(signal, "entry_price", None) or self.parameters.get("entry_price")
        if not entry_price or entry_price <= 0:
            return 0.0
        return max((portfolio_value * risk_pct) / float(entry_price), 0.0)

    def check_filters(self, data: pd.DataFrame) -> bool:
        """
        Return True if regime/market filters pass and trading is allowed.
        """
        return not data.empty

    def get_metadata(self) -> dict:
        """
        Return a dict with: name, version, description, strategy_type, symbols, parameters.
        """
        return {
            "name": self.name,
            "version": str(self.parameters.get("version", "1.0.0")),
            "description": self.parameters.get("description", self.name),
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "parameters": self.parameters,
        }

    # ── Legacy compatibility ─────────────────────────────────────────────────

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> TradeSignal:
        """
        Legacy method for backwards compatibility.
        Calls generate_signals() and wraps result in a TradeSignal.
        """
        signals = self.generate_signals(features)
        last_signal = signals.iloc[-1] if not signals.empty else Signal.HOLD
        if isinstance(last_signal, str):
            last_signal = Signal(last_signal)
        return TradeSignal(
            strategy_id=self.id,
            strategy_name=self.name,
            symbol=symbol,
            signal=last_signal,
        )

    def validate(self) -> bool:
        """Override to add parameter validation logic."""
        return bool(self.symbols)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} symbols={self.symbols}>"
