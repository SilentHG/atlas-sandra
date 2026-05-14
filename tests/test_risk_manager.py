"""
Test: Risk Manager  ─ Day 2
=============================
Tests the RiskManager against all enforced rules:
  - HOLD rejection
  - Kill switch (2 % daily loss cap)
  - Drawdown circuit-breaker
  - Per-trade risk cap (2 %)
  - Concentration limit
  - Position sizing via stop-distance
  - Daily stat reset
"""

import uuid

import pytest

from strategy_engine.base_strategy import Signal, TradeSignal
from risk_management.risk_manager import RiskManager, RiskCheckResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rm() -> RiskManager:
    """$10 000 capital, 1 % risk/trade (legacy compat), 2 % daily loss cap."""
    return RiskManager(
        capital=10_000.0,
        risk_per_trade=0.01,   # backwards-compat kwarg
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_position_pct=0.05,
    )


def _signal(
    sig: Signal,
    symbol: str = "AAPL",
    stop: float | None = 195.0,
    entry: float = 200.0,
) -> TradeSignal:
    return TradeSignal(
        strategy_id=uuid.uuid4(),
        strategy_name="test_strategy",
        symbol=symbol,
        signal=sig,
        confidence=0.8,
        entry_price=entry,
        stop_loss=stop,
        take_profit=220.0,
    )


# ── Basic approval ────────────────────────────────────────────────────────────


def test_buy_signal_approved(rm: RiskManager) -> None:
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert result.approved
    assert result.adjusted_qty is not None and result.adjusted_qty > 0


def test_sell_signal_approved(rm: RiskManager) -> None:
    result = rm.check_signal(_signal(Signal.SELL, stop=205.0), current_price=200.0)
    assert result.approved


def test_hold_signal_rejected(rm: RiskManager) -> None:
    result = rm.check_signal(_signal(Signal.HOLD), current_price=200.0)
    assert not result.approved
    assert "HOLD" in (result.rejection_reason or "")


# ── Kill switch (2 % daily loss cap) ─────────────────────────────────────────


def test_kill_switch_fires_at_2pct_loss(rm: RiskManager) -> None:
    """Record losses until the 2 % cap is hit; next order must be blocked."""
    # 2 % of $10 000 = $200; record $201 in losses
    rm.record_trade_pnl(-201.0, symbol="AAPL")
    assert rm.kill_switch_active, "Kill switch should be active after 2 % daily loss"
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert not result.approved
    assert "KILL" in (result.rejection_reason or "").upper()


def test_kill_switch_not_fired_under_cap(rm: RiskManager) -> None:
    rm.record_trade_pnl(-150.0, symbol="AAPL")  # 1.5 % — under 2 % cap
    assert not rm.kill_switch_active
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert result.approved


def test_daily_loss_legacy_circuit_breaker(rm: RiskManager) -> None:
    """Test that directly setting _daily_loss (old field) still triggers block."""
    rm._stats.gross_loss = 201.0
    rm._daily_loss       = 201.0
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert not result.approved
    assert "daily loss" in (result.rejection_reason or "").lower()


# ── Drawdown circuit-breaker ─────────────────────────────────────────────────


def test_drawdown_circuit_breaker(rm: RiskManager) -> None:
    rm.capital       = 8_500.0   # 15 % drawdown
    rm._peak_capital = 10_000.0
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert not result.approved
    assert "drawdown" in (result.rejection_reason or "").lower()


def test_drawdown_under_threshold_passes(rm: RiskManager) -> None:
    rm.capital       = 9_500.0   # 5 % drawdown — under 10 % CB
    rm._peak_capital = 10_000.0
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert result.approved


# ── Position sizing ────────────────────────────────────────────────────────────


def test_position_sizing_with_stop(rm: RiskManager) -> None:
    """risk = 1 % of $10 000 = $100 / ($200 - $195) = 20 shares."""
    result = rm.check_signal(_signal(Signal.BUY, stop=195.0), current_price=200.0)
    assert result.approved
    assert abs(result.adjusted_qty - 20.0) < 0.01


def test_position_sizing_without_stop(rm: RiskManager) -> None:
    """Fallback: risk_dollars / price."""
    result = rm.check_signal(_signal(Signal.BUY, stop=None), current_price=200.0)
    assert result.approved
    assert result.adjusted_qty is not None and result.adjusted_qty > 0


def test_per_trade_risk_capped_at_2pct() -> None:
    """With 2 % cap and a wide stop, qty should be auto-reduced."""
    rm = RiskManager(capital=10_000.0, max_risk_per_trade=0.02)
    # stop distance = $50, so uncapped qty = 200/50 = 4 shares, risk = $200 = 2 % exactly
    result = rm.check_signal(
        _signal(Signal.BUY, stop=150.0, entry=200.0), current_price=200.0
    )
    assert result.approved
    assert result.portfolio_risk_pct is not None
    assert result.portfolio_risk_pct <= 0.02 + 1e-9


# ── Concentration ─────────────────────────────────────────────────────────────


def test_concentration_limit_blocks_order(rm: RiskManager) -> None:
    """Symbol already has 5 % of capital deployed → reject."""
    result = rm.check_signal(
        _signal(Signal.BUY),
        current_price=200.0,
        symbol_open_value=500.0,   # 5 % of $10 000
    )
    assert not result.approved
    assert "concentration" in (result.rejection_reason or "").lower()


def test_concentration_under_limit_passes(rm: RiskManager) -> None:
    result = rm.check_signal(
        _signal(Signal.BUY),
        current_price=200.0,
        symbol_open_value=400.0,   # 4 % — under 5 % limit
    )
    assert result.approved


# ── Daily stats reset ─────────────────────────────────────────────────────────


def test_reset_daily_stats(rm: RiskManager) -> None:
    rm.record_trade_pnl(-150.0)
    rm.reset_daily_stats()
    assert rm.daily_loss == 0.0
    assert rm._daily_loss == 0.0
    assert not rm.kill_switch_active


def test_kill_switch_cleared_after_reset(rm: RiskManager) -> None:
    rm.record_trade_pnl(-300.0)
    assert rm.kill_switch_active
    rm.reset_daily_stats()
    assert not rm.kill_switch_active
    # Should approve again after reset
    result = rm.check_signal(_signal(Signal.BUY), current_price=200.0)
    assert result.approved


# ── P&L tracking ─────────────────────────────────────────────────────────────


def test_pnl_tracking(rm: RiskManager) -> None:
    rm.record_trade_pnl(+500.0)
    rm.record_trade_pnl(-200.0)
    assert rm.daily_pnl == pytest.approx(300.0)
    assert rm.daily_loss == pytest.approx(200.0)
    assert rm.capital    == pytest.approx(10_300.0)


def test_peak_capital_updates_on_profit(rm: RiskManager) -> None:
    rm.record_trade_pnl(+2_000.0)
    assert rm._peak_capital == pytest.approx(12_000.0)


# ── Snapshot ──────────────────────────────────────────────────────────────────


def test_snapshot_keys(rm: RiskManager) -> None:
    snap = rm.snapshot()
    required = {
        "timestamp", "capital", "peak_capital", "drawdown_pct",
        "daily_pnl", "daily_loss", "daily_loss_cap_usd",
        "trade_count", "kill_switch",
    }
    assert required.issubset(snap.keys())
    assert snap["kill_switch"] is False
    assert snap["capital"] == pytest.approx(10_000.0)
