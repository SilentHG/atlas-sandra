"""
ATLAS Base Strategy
====================
Abstract base for all trading strategies.
Concrete strategies override `generate_signal()`.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
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
    """Abstract base class for all ATLAS strategies."""

    strategy_type: str = "base"

    def __init__(self, name: str, symbols: list[str], parameters: dict[str, Any] | None = None) -> None:
        self.id         = uuid.uuid4()
        self.name       = name
        self.symbols    = symbols
        self.parameters = parameters or {}
        self.is_active  = False
        self.is_paper   = True

    @abstractmethod
    def generate_signal(self, symbol: str, features: pd.DataFrame) -> TradeSignal:
        """
        Given a DataFrame of OHLCV + features (latest row = most recent),
        return a TradeSignal.
        """

    def validate(self) -> bool:
        """Override to add parameter validation logic."""
        return bool(self.symbols)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} symbols={self.symbols}>"
