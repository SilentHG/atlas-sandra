from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from strategy_engine.base_strategy import BaseStrategy, Signal


class IndicatorRuleStrategy(BaseStrategy):
    strategy_type = "generated"

    def _col(self, data: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
        if name in data:
            return pd.to_numeric(data[name], errors="coerce")
        return pd.Series([default] * len(data), index=data.index)

    def _signals_from_rules(
        self,
        data: pd.DataFrame,
        buy_rule: pd.Series,
        sell_rule: pd.Series,
    ) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=data.index, dtype=object)
        signals.loc[buy_rule.fillna(False)] = Signal.BUY
        signals.loc[sell_rule.fillna(False)] = Signal.SELL
        return signals

    def compute_position_size(self, signal, portfolio_value: float) -> float:
        risk_pct = float(self.parameters.get("risk_pct", 0.01))
        max_pct = float(self.parameters.get("max_pct", 0.05))
        price = float(getattr(signal, "entry_price", 0) or self.parameters.get("reference_price", 100))
        notional = min(portfolio_value * risk_pct, portfolio_value * max_pct)
        return max(0.0, notional / price) if price > 0 else 0.0

    def check_filters(self, data: pd.DataFrame) -> bool:
        if data.empty:
            return False
        regime = self._col(data, "regime_trend", 1.0).iloc[-1]
        high_vol = self._col(data, "regime_high_vol", 0.0).iloc[-1]
        return bool(pd.notna(regime) and regime >= -1 and high_vol <= 1)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "version": "1.0.0",
            "description": self.parameters.get("description", self.name),
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "parameters": self.parameters,
        }


class MomentumEmaRsi(IndicatorRuleStrategy):
    strategy_type = "momentum"

    def __init__(self) -> None:
        super().__init__("momentum_ema_rsi", ["AAPL", "MSFT", "NVDA"], {"risk_pct": 0.01})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        ema9 = self._col(data, "ema_9")
        ema21 = self._col(data, "ema_21")
        rsi = self._col(data, "rsi_14")
        macd = self._col(data, "macd_hist")
        return self._signals_from_rules(data, (close > ema9) & (ema9 > ema21) & rsi.between(50, 70) & (macd > 0), (close < ema21) | (rsi > 75))


class BollingerMeanReversion(IndicatorRuleStrategy):
    strategy_type = "mean_reversion"

    def __init__(self) -> None:
        super().__init__("bollinger_mean_reversion", ["AAPL", "JPM", "V"], {"risk_pct": 0.01, "max_pct": 0.05})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        pct_b = self._col(data, "bb_pct_b")
        rsi = self._col(data, "rsi_14")
        wr = self._col(data, "williams_r_14")
        cci = self._col(data, "cci_20")
        return self._signals_from_rules(data, (pct_b < 0.1) & (rsi < 35) & (wr < -85) & (cci.abs() < 100), (pct_b > 0.5) | (rsi > 60))


class VolumeBreakout(IndicatorRuleStrategy):
    strategy_type = "breakout"

    def __init__(self) -> None:
        super().__init__("volume_breakout", ["AAPL", "MSFT", "NVDA", "AMZN"], {"risk_pct": 0.015})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        adx = self._col(data, "adx_14")
        rel_vol = self._col(data, "rel_volume_20")
        high20 = close.rolling(20).max().shift(1)
        return self._signals_from_rules(data, (close > high20) & (rel_vol > 2.0) & (adx > 25), close < close.rolling(10).mean())


class RsiDivergenceProxy(IndicatorRuleStrategy):
    strategy_type = "mean_reversion"

    def __init__(self) -> None:
        super().__init__("rsi_divergence_proxy", ["TSLA", "NVDA", "AMZN"], {"risk_pct": 0.01})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        rsi = self._col(data, "rsi_14")
        stoch_k = self._col(data, "stoch_k_14")
        stoch_d = self._col(data, "stoch_d_3")
        lower_low = close < close.rolling(10).min().shift(1)
        rsi_higher = rsi > rsi.rolling(10).min().shift(1)
        cross_up = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
        return self._signals_from_rules(data, lower_low & rsi_higher & cross_up & (stoch_k < 30), rsi > 65)


class VwapReversion(IndicatorRuleStrategy):
    strategy_type = "mean_reversion"

    def __init__(self) -> None:
        super().__init__("vwap_reversion", ["AAPL", "MSFT", "AMZN"], {"risk_pct": 0.005})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        vwap = self._col(data, "vwap")
        rsi7 = self._col(data, "rsi_7")
        return self._signals_from_rules(data, (close < vwap * 0.99) & (rsi7 < 30), (close >= vwap) | (rsi7 > 70))


class GoldenCrossMomentum(IndicatorRuleStrategy):
    strategy_type = "trend"

    def __init__(self) -> None:
        super().__init__("golden_cross_momentum", ["AAPL", "MSFT", "GOOGL"], {"risk_pct": 0.02, "max_pct": 0.10})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        macd = self._col(data, "macd_hist")
        cross = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
        return self._signals_from_rules(data, cross & (macd > 0), (sma50 < sma200) | (macd < 0))


class StochasticOversoldBounce(IndicatorRuleStrategy):
    strategy_type = "mean_reversion"

    def __init__(self) -> None:
        super().__init__("stochastic_oversold_bounce", ["JPM", "BAC", "GS"], {"risk_pct": 0.01})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        k = self._col(data, "stoch_k_14")
        d = self._col(data, "stoch_d_3")
        trend = self._col(data, "regime_trend")
        cross = (k > d) & (k.shift(1) <= d.shift(1))
        return self._signals_from_rules(data, (k < 25) & (d < 25) & cross & (trend > 0), (k > 80) | (self._col(data, "rsi_14") > 70))


class CciTrendPull(IndicatorRuleStrategy):
    strategy_type = "trend"

    def __init__(self) -> None:
        super().__init__("cci_trend_pull", ["NVDA", "TSLA", "AMD"], {"risk_pct": 0.015})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        cci = self._col(data, "cci_20")
        ema9 = self._col(data, "ema_9")
        ema21 = self._col(data, "ema_21")
        buy = (cci > 100) & (cci.shift(1) <= 100) & (ema9 > ema21) & (self._col(data, "rel_volume_20") > 1)
        return self._signals_from_rules(data, buy, (cci < 100) | (ema9 < ema21))


class WilliamsMomentum(IndicatorRuleStrategy):
    strategy_type = "momentum"

    def __init__(self) -> None:
        super().__init__("williams_momentum", ["AAPL", "MSFT", "AMZN"], {"risk_pct": 0.01})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        wr = self._col(data, "williams_r_14")
        macd = self._col(data, "macd_hist")
        close = self._col(data, "close")
        sma20 = close.rolling(20).mean()
        return self._signals_from_rules(data, (wr > -50) & (wr.shift(1) <= -80) & (macd > 0) & (close > sma20), (wr > -20) | (macd < 0))


class ObvDivergenceProxy(IndicatorRuleStrategy):
    strategy_type = "mean_reversion"

    def __init__(self) -> None:
        super().__init__("obv_divergence_proxy", ["PG", "KO", "PEP"], {"risk_pct": 0.01, "max_pct": 0.05})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close")
        obv = self._col(data, "obv")
        rsi = self._col(data, "rsi_14")
        price_lower_low = close < close.rolling(20).min().shift(1)
        obv_higher_low = obv > obv.rolling(20).min().shift(1)
        return self._signals_from_rules(data, price_lower_low & obv_higher_low & (rsi < 40), (rsi > 70) | (obv < obv.rolling(10).mean()))


GENERATED_STRATEGIES = [
    MomentumEmaRsi,
    BollingerMeanReversion,
    VolumeBreakout,
    RsiDivergenceProxy,
    VwapReversion,
    GoldenCrossMomentum,
    StochasticOversoldBounce,
    CciTrendPull,
    WilliamsMomentum,
    ObvDivergenceProxy,
]
