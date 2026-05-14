"""
Test: Risk Manager
===================
"""

import uuid
import pytest
from strategy_engine.base_strategy import TradeSignal, Signal
from risk_management.risk_manager import RiskManager


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(capital=10_000.0, risk_per_trade=0.01)


def _signal(sig: Signal, symbol: str = "AAPL", stop: float | None = 195.0) -> TradeSignal:
    return TradeSignal(
        strategy_id=uuid.uuid4(),
        strategy_name="test_strategy",
        symbol=symbol,
        signal=sig,
        confidence=0.8,
        entry_price=200.0,
        stop_loss=stop,
        take_profit=220.0,
    )


def test_buy_signal_approved(rm):
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert result.approved
    assert result.adjusted_qty is not None and result.adjusted_qty > 0


def test_hold_signal_rejected(rm):
    result = rm.check_signal(_signal(Signal.HOLD), current_price=200.0)
    assert not result.approved


def test_daily_loss_circuit_breaker(rm):
    rm._daily_loss = 501.0
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert not result.approved
    assert "daily loss" in result.rejection_reason.lower()


def test_drawdown_circuit_breaker(rm):
    rm.capital       = 8_500.0   # 15 % drawdown
    rm._peak_capital = 10_000.0
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert not result.approved
    assert "drawdown" in result.rejection_reason.lower()


def test_position_sizing_with_stop(rm):
    # risk = 1 % of 10k = $100 / ($200 - $195) = 20 shares
    result = rm.check_signal(_signal(Signal.BUY, stop=195.0), current_price=200.0)
    assert result.approved
    assert abs(result.adjusted_qty - 20.0) < 0.01


def test_reset_daily_stats(rm):
    rm._daily_loss = 300.0
    rm.reset_daily_stats()
    assert rm._daily_loss == 0.0
