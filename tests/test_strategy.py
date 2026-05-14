"""
Test: Base Strategy & Signal
=============================
"""

import uuid
import pandas as pd
import pytest

from strategy_engine.base_strategy import BaseStrategy, Signal, TradeSignal


class SimpleMAStrategy(BaseStrategy):
    """Minimal concrete strategy: buy when EMA-20 crosses above EMA-50."""

    strategy_type = "trend"

    def generate_signal(self, symbol: str, features: pd.DataFrame) -> TradeSignal:
        if len(features) < 2:
            return TradeSignal(strategy_id=self.id, strategy_name=self.name,
                               symbol=symbol, signal=Signal.HOLD)
        ema20 = features["close"].ewm(span=20).mean()
        ema50 = features["close"].ewm(span=50).mean()
        if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] <= ema50.iloc[-2]:
            return TradeSignal(strategy_id=self.id, strategy_name=self.name,
                               symbol=symbol, signal=Signal.BUY, confidence=0.75)
        return TradeSignal(strategy_id=self.id, strategy_name=self.name,
                           symbol=symbol, signal=Signal.HOLD)


@pytest.fixture
def strategy() -> SimpleMAStrategy:
    return SimpleMAStrategy(name="ma_crossover", symbols=["AAPL"])


def test_strategy_instantiation(strategy):
    assert strategy.name == "ma_crossover"
    assert "AAPL" in strategy.symbols
    assert strategy.strategy_type == "trend"


def test_hold_on_insufficient_data(strategy):
    df    = pd.DataFrame({"close": [100.0]})
    sig   = strategy.generate_signal("AAPL", df)
    assert sig.signal == Signal.HOLD


def test_signal_enum_values():
    assert Signal.BUY   == "BUY"
    assert Signal.SELL  == "SELL"
    assert Signal.HOLD  == "HOLD"
    assert Signal.CLOSE == "CLOSE"
